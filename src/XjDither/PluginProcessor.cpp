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

void XjDitherProcessor::prepareToPlay (double /*sampleRate*/, int samplesPerBlock)
{
    // Pre-allocate noise scratch buffers — never allocate in the audio thread.
    // Add a small headroom margin in case the host sends a slightly larger block.
    const int capacity = samplesPerBlock + 32;
    noiseL.resize (capacity);
    noiseR.resize (capacity);

    seedRNG (static_cast<uint64_t> (juce::Time::currentTimeMillis()));
}

void XjDitherProcessor::releaseResources()
{
    noiseL.clear();
    noiseR.clear();
    noiseL.shrink_to_fit();
    noiseR.shrink_to_fit();
}

//==============================================================================
template <typename FloatType>
void XjDitherProcessor::applyDither (juce::AudioBuffer<FloatType>& buffer)
{
    const int numChannels = buffer.getNumChannels();
    const int numSamples  = buffer.getNumSamples();

    // Safety: if the host sends a block larger than we prepared for,
    // resize on the fly. This should never happen in normal operation.
    if (numSamples > static_cast<int> (noiseL.size()))
    {
        noiseL.resize (numSamples);
        noiseR.resize (numSamples);
    }

    if (numChannels == 2)
    {
        // --- Stereo path ---
        // Step 1: fill noise buffers — scalar, sequential RNG state updates.
        fillNoise (noiseL.data(), numSamples, 0);
        fillNoise (noiseR.data(), numSamples, 1);

        FloatType* L = buffer.getWritePointer (0);
        FloatType* R = buffer.getWritePointer (1);

        // Step 2: add noise to audio.
        // These are simple independent loops with no loop-carried dependencies.
        // The compiler will auto-vectorise to SSE2 (addpd) or AVX (vaddpd)
        // depending on the target flags set in CMakeLists.
        // For double buffers this is a straight 1:1 add.
        // For float buffers the noise doubles are narrowed to float — the
        // precision loss is fine since we only need 24-bit accuracy here.
        for (int i = 0; i < numSamples; ++i)
            L[i] += static_cast<FloatType> (noiseL[i]);

        for (int i = 0; i < numSamples; ++i)
            R[i] += static_cast<FloatType> (noiseR[i]);
    }
    else
    {
        // --- Mono fallback ---
        fillNoise (noiseL.data(), numSamples, 0);

        FloatType* M = buffer.getWritePointer (0);

        for (int i = 0; i < numSamples; ++i)
            M[i] += static_cast<FloatType> (noiseL[i]);
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
