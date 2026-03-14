#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include "TransformerCore.h"

//==============================================================================
class XjTFProcessor : public juce::AudioProcessor,
                      public juce::AudioProcessorValueTreeState::Listener
{
public:
    XjTFProcessor();
    ~XjTFProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlock (juce::AudioBuffer<double>&, juce::MidiBuffer&) override;
    bool supportsDoublePrecisionProcessing() const override { return true; }

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
    void parameterChanged (const juce::String& paramID, float newValue) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    juce::AudioProcessorValueTreeState apvts;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainInputChannelSet()  != juce::AudioChannelSet::stereo()) return false;
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) return false;
        return true;
    }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    template <typename Sample>
    void processImpl (juce::AudioBuffer<Sample>& buffer);
    void prepareDSP ();

    // DSP
    juce::dsp::Oversampling<double> oversampling;

    TransformerCore transformer;

    // Parameter pointers (grabbed once in prepareToPlay)
    std::atomic<float>* driveParam   = nullptr;
    std::atomic<float>* outputParam  = nullptr;
    std::atomic<bool> needPrepare { true };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjTFProcessor)
};
