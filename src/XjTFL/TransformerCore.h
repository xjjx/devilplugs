#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <random>

class TransformerCore
{
public:
    void prepare(double sampleRate, size_t numChannels)
    {
        sr           = sampleRate;
        numCh        = numChannels;

        flux.assign   (numChannels, 0.0);
        prevMag.assign(numChannels, 0.0);

        juce::dsp::ProcessSpec spec;
        spec.sampleRate       = sampleRate;
        spec.maximumBlockSize = 512;
        spec.numChannels      = static_cast<juce::uint32>(numChannels);

        // ── Stage 1: DC blocker / input highpass ─────────────────────────
        // Removes DC offset and sub-bass before saturation (avoids asymmetric clipping)
        dcBlocker.prepare(spec);
        dcBlocker.setType(juce::dsp::StateVariableTPTFilterType::highpass);
        dcBlocker.setCutoffFrequency(5.0f);

        // ── Stage 3: Iron smoothing / output lowpass ──────────────────────
        // Models the bandwidth limiting of transformer iron core
        // 6kHz  = dark vintage iron
        // 10kHz = classic transformer
        // 16kHz = clean/modern
        ironFilter.prepare(spec);
        ironFilter.setType(juce::dsp::StateVariableTPTFilterType::lowpass);
        ironFilter.setCutoffFrequency(22000.0f);

        rng.seed(0);
    }

    // Call once after prepare() if you want to expose iron character as a param
    void setIronCutoff(float hz)
    {
        ironFilter.setCutoffFrequency(hz);
    }

    void setDrive(float d)
    {
        drive = static_cast<double>(d);
        // When drive > 1.0, phi steady state → tanh(drive * in) which is louder.
        // Dividing by drive compensates — at drive=1.0 no change, at drive=2.0 -6dB correction.
        // Use actual drive value, not tanh(drive), because tanh asymptotes to 1.0
        // and stops compensating at high drive where you actually need it most.
        driveNorm = 1.0 / std::max(drive, 0.001);
    }

    double processSample(double x, int ch)
    {
        // ── Stage 1: input highpass ───────────────────────────────────────
        double in = static_cast<double>(dcBlocker.processSample(ch, static_cast<float>(x)));

        // ── Stage 2: transformer core model ──────────────────────────────

        double& phi   = flux[static_cast<size_t>(ch)];
        double& mPrev = prevMag[static_cast<size_t>(ch)];

        // Drive input into magnetic field H
        double H = drive * in;

        // Hysteresis: magnetization M depends on H and previous state.
        // mPrev feedback models magnetic memory (remanence).
        // Kept small so it adds character without LPF — pure saturation shape.
//        double M = fastTanh(H + hysteresis * mPrev);
        double M = std::tanh(H + hysteresis * mPrev);

        // Flux integrator: phi tracks M with a one-pole lag.
        // fluxRate = 1.0 means phi = M instantly (no extra LPF).
        // Lower values add transformer "slowness" / LF boost character.
        // Keep fluxRate in (0, 1] — above 1.0 the loop overshoots.
        phi += fluxRate * (M - phi);

        // Eddy current loss: damps rapid flux changes (subtle HF softening)
        // Very small value — just enough for physical realism
        double prevPhi = phi; // store before noise
        phi -= eddy * (phi - mPrev);

        // Barkhausen noise: tiny random jumps from magnetic domain switching
        double dFlux = phi - prevPhi;
        phi += barkhausenAmount * randomNoise() * std::abs(dFlux);

        // Store previous magnetization for next sample's hysteresis
        mPrev = M;

        // Core output is phi — at steady state phi ≈ tanh(drive * in)
        // No gain compensation needed when fluxRate = 1.0
        double y = phi * driveNorm;

        // ── Stage 3: iron smoothing lowpass ───────────────────────────────
        y = static_cast<double>(ironFilter.processSample(ch, static_cast<float>(y)));

        return y;
    }

private:
    inline double fastTanh(double x)
    {
        double x2 = x * x;
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    inline double randomNoise()
    {
        return dist(rng);
    }

    double sr    = 44100.0;
    size_t numCh = 2;

    std::vector<double> flux;    // magnetic flux state per channel
    std::vector<double> prevMag; // previous magnetization per channel

    // Core model parameters — double for audio path consistency
    double drive             = 1.0;   // set via setDrive() — see exponential scaling in processor
    double driveNorm         = 1.0;   // precomputed in setDrive()
    double hysteresis        = 0.1;   // magnetic memory — keep low (0.05–0.15) to avoid LPF
    double fluxRate          = 1.0;   // 1.0 = no integrator lag; lower = more "iron slowness"
    double eddy              = 0.005; // eddy current damping — very subtle
    double barkhausenAmount  = 0.0005;// domain noise — barely audible, just adds life

    juce::dsp::StateVariableTPTFilter<float> dcBlocker;  // stage 1
    // (stage 2 is the inline core model above)
    juce::dsp::StateVariableTPTFilter<float> ironFilter; // stage 3

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{ -1.0, 1.0 };
};
