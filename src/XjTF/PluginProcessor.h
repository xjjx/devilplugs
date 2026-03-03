#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <chowdsp_wdf/chowdsp_wdf.h>

#include <chowdsp_wdf/chowdsp_wdf.h>

// Transformer WDF circuit per channel:
namespace wdft = chowdsp::wdft;
struct TransformerWDF
{
    wdft::ResistiveVoltageSourceT<double> Vs { 600.0f };
    wdft::InductorT<double> Lp { 0.02f };
    wdft::ResistorT<double> Rload { 47000.0f };

    wdft::WDFSeriesT<double, decltype(Vs), decltype(Lp)> S1 { Vs, Lp };
    wdft::WDFParallelT<double, decltype(S1), decltype(Rload)> root { S1, Rload };

    void prepare (double sampleRate)
    {
        Lp.prepare (sampleRate);
    }

    double process (double input)
    {
        Vs.setVoltage (input);
        root.incident (0.0f);
        root.reflected();
        return wdft::voltage<double> (Rload);
    }
};

// fast xorshift random, one per channel
struct NoiseGen
{
    uint32_t state = 12345;
    double next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (double) state / (double) 0xFFFFFFFF - 0.5;
    }
};

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

    std::unique_ptr<TransformerWDF> transformerWDF[2];
    std::unique_ptr<NoiseGen> noiseGen[2];

    // DC blocker per channel
    struct DCBlocker
    {
        double x1 = 0.f, y1 = 0.f;
        double process (double x, double R = 0.9998f)
        {
            double y = x - x1 + R * y1;
            x1 = x; y1 = y;
            return y;
        }
    };
    std::array<DCBlocker, 2> dcBlocker;

    // input LPF stages
    juce::dsp::ProcessorDuplicator<
        juce::dsp::IIR::Filter<double>,
        juce::dsp::IIR::Coefficients<double>> inputLPF1, inputLPF2;

    // output LPF stages
    juce::dsp::ProcessorDuplicator<
        juce::dsp::IIR::Filter<double>,
        juce::dsp::IIR::Coefficients<double>> outputLPF1, outputLPF2;

    // Frequency coloring filters
    juce::dsp::ProcessorDuplicator<
        juce::dsp::IIR::Filter<double>,
        juce::dsp::IIR::Coefficients<double>> lowShelf, highShelf;

    // Parameter pointers (grabbed once in prepareToPlay)
    std::atomic<float>* driveParam   = nullptr;
    std::atomic<float>* characterParam = nullptr;
    std::atomic<float>* toneParam    = nullptr;
    std::atomic<float>* outputParam  = nullptr;
    std::atomic<float>* saturationParam = nullptr;
    std::atomic<float>* oversamplingParam = nullptr;
    std::atomic<float>* instabilityParam = nullptr;
    std::atomic<bool> needPrepare { true };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjTFProcessor)
};
