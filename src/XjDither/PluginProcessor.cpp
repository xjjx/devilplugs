#include "PluginProcessor.h"

XjDitherProcessor::XjDitherProcessor()
    : AudioProcessor (BusesProperties()
        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
}

bool XjDitherProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    const auto& set = layouts.getMainOutputChannelSet();
    return set == juce::AudioChannelSet::mono()
        || set == juce::AudioChannelSet::stereo();
}

void XjDitherProcessor::prepareToPlay (double /*sampleRate*/, int /*samplesPerBlock*/)
{
    seedRNG (static_cast<uint64_t> (juce::Time::currentTimeMillis()));
}

//==============================================================================
template <typename FloatType>
void XjDitherProcessor::applyDither (juce::AudioBuffer<FloatType>& buffer)
{
    const int numChannels = buffer.getNumChannels();
    const int numSamples  = buffer.getNumSamples();

    if (numChannels == 2)
    {
        FloatType* L = buffer.getWritePointer (0);
        FloatType* R = buffer.getWritePointer (1);

        for (int i = 0; i < numSamples; ++i)
        {
            // Generate { noiseL, noiseR } for both channels in one SSE2 pass.
            __m128d noise = tpdfSSE2();

            // Extract scalars. _mm_cvtsd_f64 gets the low lane (Left).
            // _mm_unpackhi_pd moves the high lane (Right) to low before extracting.
            const double noiseL = _mm_cvtsd_f64 (noise);
            const double noiseR = _mm_cvtsd_f64 (_mm_unpackhi_pd (noise, noise));

            L[i] += static_cast<FloatType> (noiseL);
            R[i] += static_cast<FloatType> (noiseR);
        }
        return;
    }

    // Scalar fallback — mono or builds without SSE2.
    for (int ch = 0; ch < numChannels; ++ch)
    {
        FloatType* data = buffer.getWritePointer (ch);
        const int  rngCh = juce::jmin (ch, 1); // clamp to our two RNG states

        for (int i = 0; i < numSamples; ++i)
            data[i] += static_cast<FloatType> (tpdfScalar (rngCh));
    }
}

void XjDitherProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                       juce::MidiBuffer&)
{
    applyDither (buffer);
}

void XjDitherProcessor::processBlock (juce::AudioBuffer<double>& buffer,
                                       juce::MidiBuffer&)
{
    applyDither (buffer);
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new XjDitherProcessor();
}
