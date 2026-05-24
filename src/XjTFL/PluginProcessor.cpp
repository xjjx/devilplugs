#include "PluginProcessor.h"

//==============================================================================
// Parameter IDs
static const juce::String DRIVE_ID     = "drive";
static const juce::String OUTPUT_ID    = "output";
static const juce::String IRON_LP_ID   = "ironLP";
static const juce::String OVERSAMPLE_ID = "oversample";

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

    // Iron lowpass on/off
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        IRON_LP_ID, "Iron LP",
        true)); // on by default

    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        OVERSAMPLE_ID, "Oversampling",
        juce::StringArray { "1x", "2x", "4x" },
        0)); // default: 1x (no oversampling)

    return { params.begin(), params.end() };
}

//==============================================================================
XjTFProcessor::XjTFProcessor()
    : AudioProcessor (BusesProperties()
                        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "Parameters", createParameterLayout())
{
    apvts.addParameterListener (DRIVE_ID, this);
    apvts.addParameterListener (IRON_LP_ID, this);
    apvts.addParameterListener (OVERSAMPLE_ID, this);
}

XjTFProcessor::~XjTFProcessor()
{
    apvts.removeParameterListener (DRIVE_ID, this);
    apvts.removeParameterListener (IRON_LP_ID, this);
    apvts.removeParameterListener (OVERSAMPLE_ID, this);
}

//==============================================================================
void XjTFProcessor::parameterChanged (const juce::String& paramID, float)
{
    if ( paramID == DRIVE_ID || paramID == IRON_LP_ID || paramID == OVERSAMPLE_ID)
        needPrepare = true;
}

//==============================================================================
void XjTFProcessor::prepareDSP ()
{
    // Oversampling: rebuild with correct factor
    // 0=1x, 1=2x, 2=4x | 2^order: 2^0=1, 2^1=2, 2^2=4
    auto osParam = dynamic_cast<juce::AudioParameterChoice*>(apvts.getParameter(OVERSAMPLE_ID));
    int order = osParam->getIndex();

    oversampling = std::make_unique<juce::dsp::Oversampling<double>>(
        2,      // channels
        order,  // factor = 2^order
        juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR,
        true);

    oversampling->initProcessing(static_cast<size_t>(getBlockSize()));

    // unity (drive=1.0) at param=30
    const float drive = apvts.getParameter(DRIVE_ID)->getValue();

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

    bool ironLP = dynamic_cast<juce::AudioParameterBool*>(apvts.getParameter(IRON_LP_ID))->get();
    transformer.setIronLPEnabled(ironLP);

    transformer.setDrive(driveInternal);
}

//==============================================================================
void XjTFProcessor::prepareToPlay (double sampleRate, int /* samplesPerBlock */)
{
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

    const float outputDb  = apvts.getParameter(OUTPUT_ID)->getValue();
    const double outputGain = juce::Decibels::decibelsToGain (outputDb);

    if (needPrepare.exchange (false))
        prepareDSP ();

   if (oversampling == nullptr)
        return;

    int numChannels = buffer.getNumChannels();
    int numSamples  = buffer.getNumSamples();

    // Convert input to double
    juce::AudioBuffer<double> doubleBuffer (numChannels, numSamples);
    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i)
            doubleBuffer.setSample (ch, i, static_cast<double> (buffer.getSample (ch, i)));

    juce::dsp::AudioBlock<double> block (doubleBuffer);

    // Upsample
    auto osBlock = oversampling->processSamplesUp (block);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = osBlock.getChannelPointer (static_cast<size_t>(ch));

        for (int i = 0; i < numSamples; ++i)
            data[i] = static_cast<Sample>(transformer.processSample(data[i], ch));
    }

    // Downsample
    oversampling->processSamplesDown (block);

    // Convert back to Sample
    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i)
            buffer.setSample (ch, i, static_cast<Sample> (doubleBuffer.getSample (ch, i)));

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
