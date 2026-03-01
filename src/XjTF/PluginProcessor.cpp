#include "PluginProcessor.h"

//==============================================================================
// Parameter IDs
static const juce::String DRIVE_ID     = "drive";
static const juce::String CHARACTER_ID = "character";
static const juce::String TONE_ID      = "tone";
static const juce::String OUTPUT_ID    = "output";

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
XjTFProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Drive: 0..100 %
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        DRIVE_ID, "Drive",
        juce::NormalisableRange<float> (0.f, 100.f, 0.1f), 30.f,
        juce::AudioParameterFloatAttributes().withLabel ("%")));

    // Character: 0 = Vintage (even harmonics), 1 = Modern (more odd)
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        CHARACTER_ID, "Character",
        juce::NormalisableRange<float> (0.f, 1.f, 0.01f), 0.f,
        juce::AudioParameterFloatAttributes().withLabel ("")));

    // Tone: bass bloom / high trim, -6..+6 dB
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        TONE_ID, "Tone",
        juce::NormalisableRange<float> (-6.f, 6.f, 0.1f), 0.f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));

    // Output gain: -12..+12 dB
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        OUTPUT_ID, "Output",
        juce::NormalisableRange<float> (-12.f, 12.f, 0.1f), 0.f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));

    return { params.begin(), params.end() };
}

//==============================================================================
XjTFProcessor::XjTFProcessor()
    : AudioProcessor (BusesProperties()
                        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "Parameters", createParameterLayout()),
      oversampling (2, 2, juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true)
      // 2 channels, factor 2^2 = 4x oversampling
{
}

XjTFProcessor::~XjTFProcessor() {}

//==============================================================================
void XjTFProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Grab parameter pointers
    driveParam     = apvts.getRawParameterValue (DRIVE_ID);
    characterParam = apvts.getRawParameterValue (CHARACTER_ID);
    toneParam      = apvts.getRawParameterValue (TONE_ID);
    outputParam    = apvts.getRawParameterValue (OUTPUT_ID);

    // Prepare oversampling
    oversampling.initProcessing (static_cast<size_t> (samplesPerBlock));
    oversampling.reset();

    // Reset hysteresis & DC blocker state
    hystState.fill ({});
    dcBlocker.fill ({});

    // Prepare filters at oversampled rate
    double osRate = sampleRate * oversampling.getOversamplingFactor();
    juce::dsp::ProcessSpec spec { osRate,
                                  static_cast<uint32_t> (static_cast<size_t> (samplesPerBlock) * oversampling.getOversamplingFactor()),
                                  2 };

    *lowShelf.state  = *juce::dsp::IIR::Coefficients<float>::makeLowShelf  (osRate, 120.0,  0.7f, 1.f);
    *highShelf.state = *juce::dsp::IIR::Coefficients<float>::makeHighShelf (osRate, 8000.0, 0.7f, 1.f);
    lowShelf.prepare  (spec);
    highShelf.prepare (spec);
}

void XjTFProcessor::releaseResources()
{
    oversampling.reset();
}

//==============================================================================
float XjTFProcessor::processSampleHysteresis (float input,
                                               HysteresisState& state,
                                               float drive,
                                               float saturation)
{
    // Blend between dry input and driven input based on drive amount
    float driven = input * drive;

    // Soft saturation with state memory for subtle hysteresis character
    float output = std::tanh (driven * saturation) / saturation;

    // Small amount of state feedback for the "memory" characteristic
    // of magnetic hysteresis — without differentiating the signal
    output += 0.05f * state.prevOutput;
    output /= 1.05f; // normalize to prevent gain creep

    state.prevOutput = output;
    state.prevInput  = input;

    return output;
}

//==============================================================================
void XjTFProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;

    const float drive     = driveParam->load();
    const float character = characterParam->load();
    const float toneDb    = toneParam->load();
    const float outputDb  = outputParam->load();

    // Map drive 0..100 -> saturation parameters
    // Low drive = subtle even-harmonic color, high drive = heavier saturation
    const float driveNorm = drive / 100.f;                    // 0..1
    const float satAmount = 1.f + driveNorm * 4.f;            // 1..5 — controls knee tightness
    const float driveGain = 1.f + driveNorm * (0.5f + character * 1.5f); // more odd harmonics with character

    // Update tone filters (low shelf boost / high shelf trim tied to Tone knob)
    // Tone > 0: bass bloom up + highs slightly down. Tone < 0: reverse.
    {
        double osRate = getSampleRate() * oversampling.getOversamplingFactor();
        float lowGain  = juce::Decibels::decibelsToGain ( toneDb * 0.8f);  // ±4.8 dB bass
        float highGain = juce::Decibels::decibelsToGain (-toneDb * 0.4f);  // ±2.4 dB highs (opposite)
        *lowShelf.state  = *juce::dsp::IIR::Coefficients<float>::makeLowShelf  (osRate, 120.0,  0.7f, lowGain);
        *highShelf.state = *juce::dsp::IIR::Coefficients<float>::makeHighShelf (osRate, 8000.0, 0.7f, highGain);
    }

    const float outputGain = juce::Decibels::decibelsToGain (outputDb);

    // --- Upsample ---
    juce::dsp::AudioBlock<float> block (buffer);
    auto osBlock = oversampling.processSamplesUp (block);

    const int numChannels = static_cast<int> (osBlock.getNumChannels());
    const int numSamples  = static_cast<int> (osBlock.getNumSamples());

    for (int ch = 0; ch < numChannels; ++ch)
    {
        float* data = osBlock.getChannelPointer (static_cast<size_t> (ch));
        auto&  hs   = hystState[static_cast<size_t> (ch)];
        auto&  dc   = dcBlocker[static_cast<size_t> (ch)];

        for (int i = 0; i < numSamples; ++i)
        {
            // 1. DC block (transformers are AC-coupled)
            float s = dc.process (data[i]);

            // 2. Hysteresis / saturation
            s = processSampleHysteresis (s, hs, driveGain, satAmount);

            data[i] = s;
        }
    }

    // 3. Frequency coloring (on oversampled signal)
    juce::dsp::ProcessContextReplacing<float> ctx (osBlock);
    lowShelf.process  (ctx);
    highShelf.process (ctx);

    // --- Downsample ---
    oversampling.processSamplesDown (block);

    // 4. Output gain
    buffer.applyGain (outputGain);
}

//==============================================================================
juce::AudioProcessorEditor* XjTFProcessor::createEditor()
{
    // Use JUCE's built-in generic editor — no custom UI needed
    return new juce::GenericAudioProcessorEditor (*this);
}

//==============================================================================
void XjTFProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void XjTFProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
        apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new XjTFProcessor();
}
