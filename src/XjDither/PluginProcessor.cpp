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
	// Clamp
	sample = sample >  1.0 ?  1.0 : sample;
	sample = sample < -1.0 ? -1.0 : sample;

	// Scale to integer domain
	const double scaled = sample * kScale;

	// Truncate toward negative infinity (floor behaviour) via integer cast
	// Valid as long as |scaled| < 2^31, which is guaranteed since kScale = 2^23
	// and sample is clamped to [-1, 1]
	const auto truncated = static_cast<int32_t>(scaled);

	// Correct for negative values — integer cast truncates toward zero,
	// but floor truncates toward -infinity, so subtract 1 if we rounded up
	return static_cast<double>(truncated - (scaled < static_cast<double>(truncated) ? 1 : 0)) * kInvScale;
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
		// This __restrict__ tells the compiler that data and noisePtr
		// don't alias each other — it can then load/store them independently in SIMD registers:
		FloatType* __restrict__ data = buffer.getWritePointer(ch);
		double* __restrict__ noisePtr = noise.data();

		fillNoise(noisePtr, numSamples, ch);

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
