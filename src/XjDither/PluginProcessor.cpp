#include "PluginProcessor.h"
#include <cmath>

namespace
{
	inline double quantise24 (double sample) noexcept
	{
		// Round to nearest code, then clamp the integer code to the valid signed range
		const double q = std::floor (sample * kScale + 0.5);
		return juce::jlimit (-kScale, kScale - 1.0, q) * kInvScale;
	}

	uint64_t makeSeed (const void* self) noexcept
	{
		return static_cast<uint64_t> (juce::Random::getSystemRandom().nextInt64())
			 ^ static_cast<uint64_t> (juce::Time::currentTimeMillis())
			 ^ static_cast<uint64_t> (reinterpret_cast<uintptr_t> (self));
	}
}

XjDitherProcessor::XjDitherProcessor()
	: AudioProcessor (BusesProperties()
		.withInput	("Input",  juce::AudioChannelSet::stereo(), true)
		.withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
	seedRNG (makeSeed (this));   // valid state even if processBlock runs before prepare
}

bool XjDitherProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
	if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
		return false;

	const auto& set = layouts.getMainOutputChannelSet();
	return set == juce::AudioChannelSet::mono()
		|| set == juce::AudioChannelSet::stereo();
}

void XjDitherProcessor::prepareToPlay (double, int)
{
	seedRNG (makeSeed (this));
}

void XjDitherProcessor::releaseResources() {}

template <typename FloatType>
void XjDitherProcessor::applyDither (juce::AudioBuffer<FloatType>& buffer)
{
	constexpr int kChunk = 256;
	double noise[kChunk];

	// state[] only has two RNG streams
	const int numChannels = juce::jmin (buffer.getNumChannels(), 2);
	const int numSamples  = buffer.getNumSamples();

	for (int ch = 0; ch < numChannels; ++ch)
	{
		FloatType* data = buffer.getWritePointer (ch);

		for (int start = 0; start < numSamples; start += kChunk)
		{
			const int n = juce::jmin (kChunk, numSamples - start);

			for (int i = 0; i < n; ++i)
				noise[i] = tpdfSample (ch);

			for (int i = 0; i < n; ++i)
				data[start + i] = static_cast<FloatType> (
					quantise24 (static_cast<double> (data[start + i]) + noise[i]));
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
