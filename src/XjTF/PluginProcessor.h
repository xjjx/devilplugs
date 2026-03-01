#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <chowdsp_wdf/chowdsp_wdf.h>

#include <chowdsp_wdf/chowdsp_wdf.h>

// Transformer WDF circuit per channel:
struct TransformerWDF
{
    chowdsp::wdf::ResistiveVoltageSource<float> Vs { 600.0f };  // source with Rs built in
    chowdsp::wdf::Inductor<float> Lp { 0.02f };
    chowdsp::wdf::Resistor<float> Rload { 47000.0f };

    // Series: Vs + Lp
    chowdsp::wdf::WDFSeries<float> S1 { &Vs, &Lp };

    // Root: parallel with load — this is the termination
    chowdsp::wdf::WDFParallel<float> root { &S1, &Rload };

    void prepare (double sampleRate)
    {
        Lp.prepare (sampleRate);
    }

    float process (float input)
    {
        Vs.setVoltage (input);

        // Propagate waves — root has no parent so just call reflected
        root.reflected();
        root.incident (0.0f);  // terminated with open circuit at root

        return chowdsp::wdft::voltage<float> (Rload);
    }
};
std::array<TransformerWDF, 2> transformerWDF;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjTFProcessor)
};
