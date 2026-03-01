#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

//==============================================================================
class XjTFProcessor : public juce::AudioProcessor
{
public:
    XjTFProcessor();
    ~XjTFProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlock (juce::AudioBuffer<double>&, juce::MidiBuffer&) override {}

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; } // host-provided generic UI

    //==============================================================================
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    int getNumPrograms() override     { return 1; }
    int getCurrentProgram() override  { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    juce::AudioProcessorValueTreeState apvts;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // DSP
    juce::dsp::Oversampling<float> oversampling;

    // Per-channel hysteresis state
    struct HysteresisState
    {
        float prevInput  = 0.f;
        float prevOutput = 0.f;
    };
    std::array<HysteresisState, 2> hystState;

    // DC blocker per channel
    struct DCBlocker
    {
        float x1 = 0.f, y1 = 0.f;
        float process (float x, float R = 0.9998f)
        {
            float y = x - x1 + R * y1;
            x1 = x; y1 = y;
            return y;
        }
    };
    std::array<DCBlocker, 2> dcBlocker;

    // Frequency coloring filters
    juce::dsp::ProcessorDuplicator<
        juce::dsp::IIR::Filter<float>,
        juce::dsp::IIR::Coefficients<float>> lowShelf, highShelf;

    // Parameter pointers (grabbed once in prepareToPlay)
    std::atomic<float>* driveParam   = nullptr;
    std::atomic<float>* characterParam = nullptr;
    std::atomic<float>* toneParam    = nullptr;
    std::atomic<float>* outputParam  = nullptr;

    float processSampleHysteresis (float input, HysteresisState& state, float drive, float saturation);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjTFProcessor)
};
