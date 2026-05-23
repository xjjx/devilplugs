#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

//==============================================================================
// InputTransformer — analog input transformer emulation
// Modes: 0 = A (subtle even harmonics), 1 = S (mild odd harmonics), 2 = N (full)
//
// Signal chain per mode:
//   IN -> [pre-emphasis HF boost] -> [saturation] -> [de-emphasis HF cut] -> OUT
//
// "Movement" effects:
//   Hysteresis phase smearing  — allpass coeff modulated by signal level
//   Thermal drift              — pre/de-emphasis freq wanders via random walk
//   Core IM / transient shift  — envelope follower shifts drive point
//==============================================================================

class InputTransformerAudioProcessor : public juce::AudioProcessor
{
public:
    InputTransformerAudioProcessor();
    ~InputTransformerAudioProcessor() override = default;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&,  juce::MidiBuffer&) override;
    void processBlock(juce::AudioBuffer<double>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return false; }
    const juce::String getName() const override { return "InputTransformer"; }
    bool acceptsMidi()  const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms()    override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    bool supportsDoublePrecisionProcessing() const override { return true; }

    //==============================================================================
    juce::AudioProcessorValueTreeState apvts;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainInputChannelSet()  != juce::AudioChannelSet::stereo()) return false;
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) return false;
        return true;
    }

private:
    template <typename Sample>
    void processImpl(juce::AudioBuffer<Sample>& buffer);

    //==============================================================================
    // Per-mode DSP state — same layout for all three modes
    struct ModeState
    {
        // Pre/de-emphasis LP integrators
        double preL = 0.0, preR = 0.0;

        // Hysteresis — 1-pole allpass state
        double apL  = 0.0, apR  = 0.0;

        // Core IM — transient envelope follower
        double envL = 0.0, envR = 0.0;

        // DC blocker
        double dcL   = 0.0, dcR   = 0.0;
        double dcHpL = 0.0, dcHpR = 0.0;
    };

    ModeState modeA, modeS, modeN;

    // Mode N only — 2-pole LF resonator state
    struct ModeNExtra
    {
        double lf1L = 0.0, lf2L = 0.0;
        double lf1R = 0.0, lf2R = 0.0;

        // Thermal drift — LF resonator freq random walk state
        double driftPhase    = 0.0;   // random walk current value
        double driftVelocity = 0.0;   // random walk momentum
        double driftSmoothed = 0.0;   // smoothed output
    } modeNExtra;

    // Thermal drift — pre-emphasis freq random walk (shared A/S/N)
    struct ThermalDrift
    {
        double phase    = 0.0;
        double velocity = 0.0;
        double smoothed = 0.0;
    };
    ThermalDrift driftA, driftS, driftN;

    //==============================================================================
    // Filter coefficients (computed in prepareToPlay)
    struct Coeffs
    {
        // Pre/de-emphasis nominal shelf coefficients
        double a_pre  = 0.0;   // Mode A ~18kHz
        double s_pre  = 0.0;   // Mode S ~15kHz
        double n_pre  = 0.0;   // Mode N ~10kHz

        // Mode N LF resonator biquad (Transposed Direct Form II)
        double n_lf_b0 = 0.0, n_lf_b1 = 0.0, n_lf_b2 = 0.0;
        double n_lf_a1 = 0.0, n_lf_a2 = 0.0;

        // Hysteresis allpass — base coeff + modulation depth per mode
        // Actual coeff = base + envMono * depth (computed per sample)
        double a_apBase = 0.0, a_apDepth = 0.0;
        double s_apBase = 0.0, s_apDepth = 0.0;
        double n_apBase = 0.0, n_apDepth = 0.0;

        // Thermal drift — random walk step size and smoothing
        double driftStep   = 0.0;   // how far it can wander per sample
        double driftSmooth = 0.0;   // 1-pole smoothing coeff
        double driftRange  = 0.0;   // max fractional deviation of shelf freq

        // Core IM — envelope follower release coefficients per mode
        double a_coreRls = 0.0;   // Mode A ~80ms
        double s_coreRls = 0.0;   // Mode S ~120ms
        double n_coreRls = 0.0;   // Mode N ~180ms

        // DC blocker (shared)
        double dc = 0.0;
    } coeffs;

    double sampleRate = 44100.0;

    //==============================================================================
    // Compute pre-emphasis coeff from nominal + thermal drift offset
    // drift is in range -1..+1, driftRange is fractional (e.g. 0.08 = ±8%)
    static forcedinline double driftedCoeff(double nominal,
                                            double drift,
                                            double range,
                                            double sr) noexcept
    {
        const double pi2  = juce::MathConstants<double>::twoPi;
        // Recover nominal frequency from coeff, apply drift, recompute
        const double nomFreq     = -std::log(nominal) * sr / pi2;
        const double driftedFreq = nomFreq * (1.0 + drift * range);
        return std::exp(-pi2 * driftedFreq / sr);
    }

    // Pre-emphasis: 1-pole high shelf boost
    static forcedinline double preEmphasis(double x, double& s,
                                           double coeff, double amount) noexcept
    {
        s        += (1.0 - coeff) * (x - s);
        double hp = x - s;
        return s + hp * (1.0 + amount);
    }

    // De-emphasis: pass the saturated signal + the LP state from pre-emphasis
    // We reconstruct HP from (x - lp) using the SAME lp state
    static forcedinline double deEmphasis(double x, double lp, double amount) noexcept
    {
        // lp is the pre-emphasis integrator value — no & reference, read-only
        double hp = x - lp;                // same split point as pre-emphasis
        return lp + hp / (1.0 + amount);   // attenuate HF by exact inverse
    }

    // Hysteresis — 1-pole allpass, coeff modulated by signal level
    // coeff near 0 = near flat phase; coeff near 1 = heavy phase shift at LF
    static forcedinline double allpass1(double x, double& s, double coeff) noexcept
    {
        double y = coeff * (x - s) + s;  // 1-pole allpass
        s = y;
        return y;
    }

    // Mode A saturation: soft even-harmonic, nearly transparent
    static forcedinline double satModeA(double x, double drive) noexcept
    {
        double d = x * drive;
        double y = d - (d * d * d) * 0.04 + (d * d) * 0.008;
        return y / drive;
    }

    // Mode S saturation: odd-harmonic, forward/punchy
    static forcedinline double satModeS(double x, double drive) noexcept
    {
        double d    = x * drive;
        double sign = d >= 0.0 ? 1.0 : -1.0;
        double y    = sign * (1.0 - std::exp(-std::abs(d) * 2.5)) / 2.5;
        return y / drive;
    }

    // Mode N saturation: aggressive tanh + slight 2nd harmonic asymmetry
    static forcedinline double satModeN(double x, double drive) noexcept
    {
        double d = x * drive;
        double y = std::tanh(d * 1.6) / 1.6;
        y       += d * d * 0.018;
        return y / drive;
    }

    // Biquad Transposed Direct Form II
    static forcedinline double biquadTDF2(double x,
                                          double& s1, double& s2,
                                          double b0, double b1, double b2,
                                          double a1, double a2) noexcept
    {
        double y = b0 * x + s1;
        s1       = b1 * x - a1 * y + s2;
        s2       = b2 * x - a2 * y;
        return y;
    }

    // DC blocker
    static forcedinline double dcBlock(double x,
                                       double& s1, double& s2,
                                       double coeff) noexcept
    {
        double hp = x - s1 + coeff * s2;
        s1 = x;
        s2 = hp;
        return hp;
    }

    // Thermal drift — random walk step
    // Returns smoothed value in range -1..+1
    static forcedinline double thermalStep(ThermalDrift& d,
                                           double step,
                                           double smooth,
                                           double rnd) noexcept
    {
        // Brownian motion with soft boundaries
        d.velocity += rnd * step;
        d.velocity *= 0.995;                          // friction — prevents runaway
        d.phase    += d.velocity;
        d.phase     = juce::jlimit(-1.0, 1.0, d.phase);
        d.smoothed += (1.0 - smooth) * (d.phase - d.smoothed);
        return d.smoothed;
    }

    // Core IM — envelope follower (peak detect, variable release)
    static forcedinline double coreEnvelope(double x, double& env,
                                            double release) noexcept
    {
        double peak = std::abs(x);
        env = peak > env ? peak : env * release;
        return env;
    }

    //==============================================================================
    // Simple LCG for per-sample thermal noise — no heap, no juce::Random overhead
    uint32_t _lcg { 0x12345678u };

    forcedinline double lcgRand() noexcept
    {
        _lcg = _lcg * 1664525u + 1013904223u;
        // Map uint to -1..+1
        return static_cast<double>(static_cast<int32_t>(_lcg)) / 2147483648.0;
    }

    //==============================================================================
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InputTransformerAudioProcessor)
};
