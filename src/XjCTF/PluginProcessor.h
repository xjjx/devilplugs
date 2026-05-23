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
    // Mode A state
    struct ModeAState
    {
        double preL = 0.0, preR = 0.0;   // pre-emphasis LP integrator
        double deL  = 0.0, deR  = 0.0;   // de-emphasis LP integrator
        double dcL  = 0.0, dcR  = 0.0;   // DC blocker
        double dcHpL = 0.0, dcHpR = 0.0;
    } modeA;

    // Mode S state
    struct ModeSState
    {
        double preL = 0.0, preR = 0.0;
        double deL  = 0.0, deR  = 0.0;
        double dcL  = 0.0, dcR  = 0.0;
        double dcHpL = 0.0, dcHpR = 0.0;
    } modeS;

    // Mode N state — adds 2-pole resonant LF bump
    struct ModeNState
    {
        double preL = 0.0, preR = 0.0;
        double deL  = 0.0, deR  = 0.0;
        // LF resonator (2-pole biquad, Transposed Direct Form II)
        double lf1L = 0.0, lf2L = 0.0;
        double lf1R = 0.0, lf2R = 0.0;
        // DC blocker
        double dcL  = 0.0, dcR  = 0.0;
        double dcHpL = 0.0, dcHpR = 0.0;
    } modeN;

    //==============================================================================
    // Filter coefficients (computed in prepareToPlay)
    struct Coeffs
    {
        // Pre/de-emphasis shelf transition frequencies
        double a_pre  = 0.0;   // Mode A ~18kHz
        double s_pre  = 0.0;   // Mode S ~15kHz
        double n_pre  = 0.0;   // Mode N ~10kHz

        // Pre/de-emphasis drive amounts (scaled in processImpl from Drive param)
        // Stored here for convenience — recomputed each block
        double a_emph = 0.0;
        double s_emph = 0.0;
        double n_emph = 0.0;

        // Mode N LF resonator biquad (Transposed Direct Form II)
        double n_lf_b0 = 0.0, n_lf_b1 = 0.0, n_lf_b2 = 0.0;
        double n_lf_a1 = 0.0, n_lf_a2 = 0.0;

        // DC blocker (shared coeff)
        double dc = 0.0;
    } coeffs;

    double sampleRate = 44100.0;

    //==============================================================================
    // Pre-emphasis: 1-pole high shelf boost
    // Splits signal into LP + HP, boosts HP by amount before saturation
    static forcedinline double preEmphasis(double x, double& s,
                                           double coeff, double amount) noexcept
    {
        s        += (1.0 - coeff) * (x - s);
        double hp = x - s;
        return s + hp * (1.0 + amount);
    }

    // De-emphasis: exact inverse of preEmphasis — restores flat response
    static forcedinline double deEmphasis(double x, double& s,
                                          double coeff, double amount) noexcept
    {
        s        += (1.0 - coeff) * (x - s);
        double hp = x - s;
        return s + hp / (1.0 + amount);
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

    //==============================================================================
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InputTransformerAudioProcessor)
};
