#include "PluginProcessor.h"

//==============================================================================
// Parameter IDs
static const juce::String DRIVE_ID     = "drive";
static const juce::String CHARACTER_ID = "character";
static const juce::String TONE_ID      = "tone";
static const juce::String OUTPUT_ID    = "output";
static const juce::String SATURATION_ID = "saturation";
static const juce::String OVERSAMPLING_ID = "oversampling";
static const juce::String INSTABILITY_ID = "instability";

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
XjTFProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Saturation
    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        "saturation", "Saturation",
        juce::StringArray { "Algebraic", "Cubic", "Arctangent", "Exponential", "Asymmetric", "Tanh" },
        0));

    // Drive: 0..100 %
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        DRIVE_ID, "Drive",
        juce::NormalisableRange<float> (0.f, 100.f, 0.1f), 30.f,
        juce::AudioParameterFloatAttributes().withLabel ("%")));

    // Character: 0 = Vintage (even harmonics), 1 = Modern (more odd)
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        CHARACTER_ID, "Character",
        juce::NormalisableRange<float> (0.f, 1.f, 0.01f), 0.4f,
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

    // Instability
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        "instability", "Instability",
        juce::NormalisableRange<float> (0.f, 1.f, 0.01f), 0.f,
        juce::AudioParameterFloatAttributes().withLabel ("%")));

    // Oversampling
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        "oversampling", "Oversampling", true));

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
    transformerWDF[0] = std::make_unique<TransformerWDF>();
    transformerWDF[1] = std::make_unique<TransformerWDF>();
    noiseGen[0] = std::make_unique<NoiseGen>();
    noiseGen[1] = std::make_unique<NoiseGen>();

    apvts.addParameterListener (TONE_ID, this);
    saturationParam = apvts.getRawParameterValue (SATURATION_ID);
}

XjTFProcessor::~XjTFProcessor()
{
    apvts.removeParameterListener (TONE_ID, this);
}

//==============================================================================
void XjTFProcessor::parameterChanged (const juce::String& paramID, float)
{
    if (paramID == TONE_ID)
        toneChanged = true;
}

//==============================================================================
void XjTFProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Grab parameter pointers
    driveParam     = apvts.getRawParameterValue (DRIVE_ID);
    characterParam = apvts.getRawParameterValue (CHARACTER_ID);
    toneParam      = apvts.getRawParameterValue (TONE_ID);
    outputParam    = apvts.getRawParameterValue (OUTPUT_ID);
    instabilityParam = apvts.getRawParameterValue (INSTABILITY_ID);

    // Prepare oversampling
    oversamplingParam = apvts.getRawParameterValue (OVERSAMPLING_ID);
    oversampling.reset();
    oversampling.initProcessing (static_cast<size_t> (samplesPerBlock));

    double osRate = sampleRate * oversampling.getOversamplingFactor();
    for (int i = 0; i < 2; ++i)
        transformerWDF[i]->prepare (osRate);

    // Reset hysteresis & DC blocker state
    dcBlocker.fill ({});

    // Prepare filters at oversampled rate
    juce::dsp::ProcessSpec spec { osRate,
                                  static_cast<uint32_t> (static_cast<size_t> (samplesPerBlock) * oversampling.getOversamplingFactor()),
                                  2 };

    *lowShelf.state  = *juce::dsp::IIR::Coefficients<double>::makeLowShelf  (osRate, 120.0,  0.7, 1.0);
    *highShelf.state = *juce::dsp::IIR::Coefficients<double>::makeHighShelf (osRate, 8000.0, 0.7, 1.0);
    lowShelf.prepare  (spec);
    highShelf.prepare (spec);
}

void XjTFProcessor::releaseResources()
{
    oversampling.reset();
}

//==============================================================================
static inline double saturate (double s, double satAmount, int mode) noexcept
{
    switch (mode)
    {
        case 1: return s - (s * s * s) / (3.0 * (1.0 + std::abs (s)));
        case 2: return std::atan (s * satAmount) / (satAmount * (M_PI / 2.0));
        case 3: return std::copysign (1.0 - std::exp (-std::abs (s * satAmount)), s) / satAmount;
        case 4: return s / (1.0 + std::abs (s * satAmount) + 0.1 * s * satAmount);
        case 5: return std::tanh (s * satAmount) / satAmount;
        default: return s / (1.0 + std::abs (s * satAmount));
    }
}

template <typename Sample>
void XjTFProcessor::processImpl (juce::AudioBuffer<Sample>& buffer)
{
    juce::ScopedNoDenormals noDenormals;

    const float drive     = driveParam->load();
    const float character = characterParam->load();
    const float toneDb    = toneParam->load();
    const float outputDb  = outputParam->load();
    const float instability = instabilityParam->load();

    // Map drive 0..100 -> saturation parameters
    // Low drive = subtle even-harmonic color, high drive = heavier saturation
    const double driveNorm = drive / 100.0;                    // 0..1
    const double satAmount = 1.0 + (driveNorm * driveNorm) * 2.0;        // 1..3, gentler knee
    const double driveGain = 1.0 + (driveNorm * driveNorm) * (0.3 + character * 0.8); // much gentler

    // Update tone filters (low shelf boost / high shelf trim tied to Tone knob)
    // Tone > 0: bass bloom up + highs slightly down. Tone < 0: reverse.
    if (toneChanged.exchange (false))
    {
        lastToneDb = toneDb;
        double osRate   = getSampleRate() * oversampling.getOversamplingFactor();
        double lowGain  = juce::Decibels::decibelsToGain ((double) toneDb *  0.8);
        double highGain = juce::Decibels::decibelsToGain ((double) toneDb * -0.4);
        *lowShelf.state  = *juce::dsp::IIR::Coefficients<double>::makeLowShelf  (osRate, 120.0, 0.7, lowGain);
        *highShelf.state = *juce::dsp::IIR::Coefficients<double>::makeHighShelf (osRate, 8000.0, 0.7, highGain);
    }
    const double outputGain = juce::Decibels::decibelsToGain (outputDb);

    // --- Upsample ---
    // Convert input to double if needed
    juce::AudioBuffer<double> doubleBuffer (buffer.getNumChannels(), buffer.getNumSamples());
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        for (int i = 0; i < buffer.getNumSamples(); ++i)
            doubleBuffer.setSample (ch, i, (double) buffer.getSample (ch, i));

    juce::dsp::AudioBlock<double> block (doubleBuffer);
	const bool useOversampling = oversamplingParam->load() > 0.5f;
	auto osBlock = useOversampling ? oversampling.processSamplesUp (block) : block;

    const int numChannels = static_cast<int> (osBlock.getNumChannels());
    const int numSamples  = static_cast<int> (osBlock.getNumSamples());

    for (int ch = 0; ch < numChannels; ++ch)
    {
        double* data = osBlock.getChannelPointer (static_cast<size_t> (ch));
        auto&  dc   = dcBlocker[static_cast<size_t> (ch)];

        for (int i = 0; i < numSamples; ++i)
        {
            // 1. DC block (transformers are AC-coupled)
            double s = dc.process (data[i]);

            // Drive into transformer
            s *= driveGain;

            // WDF transformer model
            s = transformerWDF[static_cast<size_t>(ch)]->process (s);

            // Soft clip for extreme drive levels
            if (instability > 0.0)
            {
                // stochastic resonance: noise before saturator
                double noise = noiseGen[static_cast<size_t> (ch)]->next();
                s += noise * instability * 0.002;  // very subtle noise floor

                // saturation jitter: random variation of knee per sample
                double jitter = 1.0 + noiseGen[static_cast<size_t> (ch)]->next()
                            * instability * 0.05;
                s = saturate (s, satAmount * jitter, static_cast<int> (saturationParam->load()));
            }
            else
            {
                s = saturate (s, satAmount, static_cast<int> (saturationParam->load()));
            }

            data[i] = s;
        }
    }

    // 3. Frequency coloring (on oversampled signal)
    juce::dsp::ProcessContextReplacing<double> ctx (osBlock);
    lowShelf.process  (ctx);
    highShelf.process (ctx);

    // --- Downsample ---
    if (useOversampling)
        oversampling.processSamplesDown (block);

    // 4. Output gain
    doubleBuffer.applyGain (outputGain);

    // Convert back to Sample precision
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        for (int i = 0; i < buffer.getNumSamples(); ++i)
            buffer.setSample (ch, i, (Sample) doubleBuffer.getSample (ch, i));
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
