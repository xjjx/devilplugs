#include "PluginProcessor.h"

XjDitherProcessor::XjDitherProcessor()
	: AudioProcessor (BusesProperties()
		.withInput	("Input",  juce::AudioChannelSet::stereo(), true)
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
	const size_t capacity = static_cast<std::size_t>(samplesPerBlock + 32);
	noise.resize (capacity);

	seedRNG (static_cast<uint64_t> (juce::Time::currentTimeMillis()));
}

void XjDitherProcessor::releaseResources()
{
	noise.clear();
	noise.shrink_to_fit();
}

// Scale factor: 2^23 for 24-bit (signed range -2^23 to 2^23-1)
static constexpr int	kBitDepth = 24;
static constexpr double kScale	  = static_cast<double>(1 << (kBitDepth - 1));
static constexpr double kInvScale = 1.0 / static_cast<double>(1 << (kBitDepth - 1));

inline double quantise24 (double sample) noexcept
{
	// Clamp to valid range first to avoid overflow on the floor()
	sample = juce::jlimit (-1.0, 1.0, sample);

	// Scale to integer domain, truncate, scale back
	return std::floor (sample * kScale) * kInvScale;
}

//==============================================================================
template <typename FloatType>
void XjDitherProcessor::applyDither (juce::AudioBuffer<FloatType>& buffer)
{
	const int numChannels = buffer.getNumChannels();
	const size_t numSamples = static_cast<size_t>(buffer.getNumSamples());

	// Safety: if the host sends a block larger than we prepared for,
	// resize on the fly. This should never happen in normal operation.
	if (numSamples > noise.size())
		noise.resize (numSamples);

	for (int ch = 0; ch < numChannels; ++ch)
	{
		FloatType* data = buffer.getWritePointer(ch);
		double* noisePtr = noise.data();

		fillNoise(noisePtr, static_cast<int>(numSamples), ch);

		// Tight, vectorizable loop
		for (size_t i = 0; i < numSamples; ++i)
		{
			const double x = static_cast<double>(data[i]) + noisePtr[i];
			data[i] = static_cast<FloatType>(quantise24(x));
		}
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
