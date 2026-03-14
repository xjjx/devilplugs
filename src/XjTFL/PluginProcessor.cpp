#include "PluginProcessor.h"

//==============================================================================
// Parameter IDs
static const juce::String DRIVE_ID     = "drive";
static const juce::String OUTPUT_ID    = "output";

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
XjTFProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Drive: 0..100 %
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        DRIVE_ID, "Drive",
        juce::NormalisableRange<float> (0.f, 100.f, 0.1f), 50.f,
        juce::AudioParameterFloatAttributes().withLabel ("%")));

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
      oversampling (2, 2, juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR, true)
      // 2 channels, factor 2^2 = 4x oversampling
{


    apvts.addParameterListener (DRIVE_ID, this);
}

XjTFProcessor::~XjTFProcessor()
{
    apvts.removeParameterListener (DRIVE_ID, this);
}

//==============================================================================
void XjTFProcessor::parameterChanged (const juce::String& paramID, float)
{
    if ( paramID == DRIVE_ID )
        needPrepare = true;
}

//==============================================================================
void XjTFProcessor::prepareDSP ()
{
    // unity (drive=1.0) at param=30
    float drive = driveParam->load();

    float driveInternal;
    if (drive < 60.0f)
        driveInternal = std::pow(10.0f, (drive / 60.0f - 1.0f) * 2.0f);
    else
    {
        // 60 → 1.0, 100 → same as old 70
        // old 70: normalized=(70-60)/40=0.25, pow(10, 0.25*2.5) = pow(10, 0.625) ≈ 4.22
        const float maxDrive = std::pow(10.0f, 0.625f); // ≈ 4.22, what param=70 used to be
        float t = (drive - 60.0f) / 40.0f;              // 0..1
        driveInternal = std::pow(maxDrive, t);           // 1.0 → 4.22
    }

    transformer.setDrive(driveInternal);
}

//==============================================================================
void XjTFProcessor::prepareToPlay (double sampleRate, int /* samplesPerBlock */)
{
    // Grab parameter pointers
    driveParam     = apvts.getRawParameterValue (DRIVE_ID);
    outputParam    = apvts.getRawParameterValue (OUTPUT_ID);

	transformer.prepare(sampleRate, static_cast<size_t>(getTotalNumOutputChannels()));

    prepareDSP ();
}

void XjTFProcessor::releaseResources()
{
    oversampling.reset();
}

//==============================================================================

template <typename Sample>
void XjTFProcessor::processImpl (juce::AudioBuffer<Sample>& buffer)
{
    juce::ScopedNoDenormals noDenormals;

    const float outputDb  = outputParam->load();
    const double outputGain = juce::Decibels::decibelsToGain (outputDb);

    if (needPrepare.exchange (false))
        prepareDSP ();

    int numChannels = buffer.getNumChannels();
    int numSamples  = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        Sample* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i)
        {
            data[i] = transformer.processSample(data[i], ch);
        }
    }

    buffer.applyGain (outputGain);
}

void XjTFProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& /*midiMessages*/)
{
    processImpl (buffer);
}

void XjTFProcessor::processBlock (juce::AudioBuffer<double>& buffer,
                                   juce::MidiBuffer& /*midiMessages*/)
{
    processImpl (buffer);
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
