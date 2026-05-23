#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

//==============================================================================
// InputTransformer — analog input transformer emulation
// Modes: 0 = A (subtle even harmonics), 1 = S (mild odd harmonics), 2 = N (full)
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
    bool hasEditor() const override { return true; }
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
    // Mode A state — 1-pole even-harmonic sat + gentle HF rolloff
    struct SState
    {
        double hfL = 0.0, hfR = 0.0;       // HF rolloff integrator
        double dcL = 0.0, dcR = 0.0;       // DC blocker state
        double dcHpL = 0.0, dcHpR = 0.0;
    } mode_s;

    // Mode S state — odd-harmonic sat + HF rolloff (no LF bump)
    struct AState
    {
        double hfL = 0.0, hfR = 0.0;
        double dcL = 0.0, dcR = 0.0;
        double dcHpL = 0.0, dcHpR = 0.0;
    } mode_a;

    // Mode N state — 2-pole resonant LF bump + aggressive even sat + strong HF rolloff
    struct NState
    {
        // LF resonator (2-pole)
        double lf1L = 0.0, lf2L = 0.0;
        double lf1R = 0.0, lf2R = 0.0;
        // HF rolloff
        double hfL = 0.0, hfR = 0.0;
        // HF shelf cut (extra bandwidth limiting)
        double shelfL = 0.0, shelfR = 0.0;
        // DC blocker
        double dcL = 0.0, dcR = 0.0;
        double dcHpL = 0.0, dcHpR = 0.0;
    } mode_n;

    //==============================================================================
    // Filter coefficients (computed in prepareToPlay)
    struct Coeffs
    {
        // Mode S
        double mode_s_hf    = 0.0;     // HF rolloff coeff (~18kHz)
        double mode_s_dc    = 0.0;     // DC blocker

        // Mode A
        double mode_a_hf    = 0.0;     // HF rolloff coeff (~15kHz)
        double mode_a_dc    = 0.0;

        // Mode N
        double mode_n_lf_a1 = 0.0, mode_n_lf_a2 = 0.0;  // 2-pole LF resonator
        double mode_n_lf_b0 = 0.0, mode_n_lf_b1 = 0.0, mode_n_lf_b2 = 0.0;
        double mode_n_hf    = 0.0;    // HF rolloff (~12kHz)
        double mode_n_shelf = 0.0;    // extra HF shelf
        double mode_n_dc    = 0.0;
    } coeffs;

    double sampleRate = 44100.0;

    //==============================================================================
    // Saturation functions

    // Mode A: soft even-harmonic — gentle x - x^3 curve, nearly transparent
    static forcedinline double sSat(double x, double drive)
    {
        double d = x * drive;
        // Even harmonics via asymmetric bias, very subtle
        double y = d - (d * d * d) * 0.04 + (d * d) * 0.008;
        return y / drive;
    }

    // Mode S: odd-harmonic — asymmetric exponential, more forward/punchy
    static forcedinline double aSat(double x, double drive)
    {
        double d = x * drive;
        // Odd harmonics: sign-preserving saturator
        double sign = d >= 0.0 ? 1.0 : -1.0;
        double y    = sign * (1.0 - std::exp(-std::abs(d) * 2.5)) / 2.5;
        return y / drive;
    }

    // Mode N: aggressive even-harmonic tanh with slight asymmetry
    static forcedinline double nSat(double x, double drive)
    {
        double d = x * drive;
        // tanh core (even+odd) + small asymmetric 2nd harmonic push
        double y = std::tanh(d * 1.6) / 1.6;
        y       += d * d * 0.018;   // 2nd harmonic bias
        return y / drive;
    }

    // DC blocker (same for all modes)
    static forcedinline double dcBlock(double x, double& s1, double& s2, double coeff)
    {
        double hp = x - s1 + coeff * s2;
        s1 = x;
        s2 = hp;
        return hp;
    }

    //==============================================================================
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InputTransformerAudioProcessor)
};
