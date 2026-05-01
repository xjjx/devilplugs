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
	const size_t numSamples = static_cast<size_t>(buffer.getNumSamples());

	// Safety: if the host sends a block larger than we prepared for,
	// resize on the fly. This should never happen in normal operation.
	if (numSamples > noiseL.size()) {
		noiseL.resize (numSamples);
		noiseR.resize (numSamples);
	}

	if (numChannels == 2)
	{
		// --- Stereo path ---
		// Fill both noise buffers upfront so each loop below is a clean
		// independent read+add+write — easier for the compiler to vectorise.
		fillNoise (noiseL.data(), numSamples, 0);
		fillNoise (noiseR.data(), numSamples, 1);
 
		FloatType* dataL = buffer.getWritePointer (0);
		FloatType* dataR = buffer.getWritePointer (1);
 
		for (size_t i = 0; i < numSamples; ++i)
			dataL[i] = static_cast<FloatType> (quantise24 (static_cast<double> (dataL[i]) + noiseL[i]));
 
		for (size_t i = 0; i < numSamples; ++i)
			dataR[i] = static_cast<FloatType> (quantise24 (static_cast<double> (dataR[i]) + noiseR[i]));
	} else {
		// --- Mono path ---
		fillNoise (noiseL.data(), numSamples, 0);
 
		FloatType* dataM = buffer.getWritePointer (0);
 
		for (size_t i = 0; i < numSamples; ++i)
			dataM[i] = static_cast<FloatType> (quantise24 (static_cast<double> (dataM[i]) + noiseL[i]));
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
