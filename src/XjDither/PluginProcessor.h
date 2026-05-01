#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <cstdint>
#include <vector>

// 24-bit quantisation constants
static constexpr int	kBitDepth = 24;
static constexpr double kScale	  = static_cast<double> (1 << (kBitDepth - 1));
static constexpr double kInvScale = 1.0 / static_cast<double> (1 << (kBitDepth - 1));

class XjDitherProcessor : public juce::AudioProcessor
{
public:
	XjDitherProcessor();
	~XjDitherProcessor() override = default;

	//==============================================================================
	void prepareToPlay (double sampleRate, int samplesPerBlock) override;
	void releaseResources() override;
	void processBlock (juce::AudioBuffer<float>&,  juce::MidiBuffer&) override;
	void processBlock (juce::AudioBuffer<double>&, juce::MidiBuffer&) override;

	//==============================================================================
	juce::AudioProcessorEditor* createEditor() override { return nullptr; }
	bool hasEditor() const override { return false; }

	//==============================================================================
	const juce::String getName() const override { return "XjDither"; }
	bool   acceptsMidi()  const override { return false; }
	bool   producesMidi() const override { return false; }
	bool   isMidiEffect() const override { return false; }
	double getTailLengthSeconds() const override { return 0.0; }

	//==============================================================================
	int  getNumPrograms()	 override { return 1; }
	int  getCurrentProgram() override { return 0; }
	void setCurrentProgram (int) override {}
	const juce::String getProgramName (int) override { return "Default"; }
	void changeProgramName (int, const juce::String&) override {}

	//==============================================================================
	void getStateInformation (juce::MemoryBlock&) override {}
	void setStateInformation (const void*, int) override {}

	//==============================================================================
	bool supportsDoublePrecisionProcessing() const override { return true; }
	bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

private:
	// xoshiro256** state.
	// state[0..3] = Left / Mono, state[4..7] = Right.
	uint64_t state[8] {};

	//==============================================================================
	// Noise scratch buffer — allocated once in prepareToPlay.
	std::vector<double> noise;

	//==============================================================================
	static uint64_t splitmix64 (uint64_t& x) noexcept
	{
		x += 0x9e3779b97f4a7c15ULL;
		x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
		x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
		return x ^ (x >> 31);
	}

	void seedRNG (uint64_t seed) noexcept
	{
		uint64_t s = seed;
		for (int i = 0; i < 4; ++i)
			state[i] = splitmix64 (s);

		s = seed ^ 0xdeadbeefcafe1234ULL;
		for (int i = 4; i < 8; ++i)
			state[i] = splitmix64 (s);
	}

	static inline uint64_t rotl64 (uint64_t x, int k) noexcept
	{
		return (x << k) | (x >> (64 - k));
	}

	static inline uint64_t xoshiroNext (uint64_t* base) noexcept
	{
		const uint64_t result = rotl64 (base[1] * 5, 7) * 9;
		const uint64_t t	  = base[1] << 17;
		base[2] ^= base[0];
		base[3] ^= base[1];
		base[1] ^= base[2];
		base[0] ^= base[3];
		base[2] ^= t;
		base[3]  = rotl64 (base[3], 45);
		return result;
	}

	static inline double toDouble (uint64_t raw) noexcept
	{
		// Scaling factor: top 53 bits of uint64 -> double in [0, 1)
		static constexpr double kU64ToDouble = 1.0 / 9007199254740992.0;
		return static_cast<double> (raw >> 11) * kU64ToDouble * 2.0 - 1.0;
	}

	inline double tpdfSample (int ch) noexcept
	{
		uint64_t* base = state + (ch * 4);
		const double a = toDouble (xoshiroNext (base));
		const double b = toDouble (xoshiroNext (base));
		// One 24-bit LSB in normalised float domain: 1.0 / 2^23
		return (a + b) * (0.5 * kInvScale);
	}

	void fillNoise (double* dst, int n, int ch) noexcept
	{
		for (int i = 0; i < n; ++i)
			dst[i] = tpdfSample (ch);
	}

	//==============================================================================
	template <typename FloatType>
	void applyDither (juce::AudioBuffer<FloatType>& buffer);

	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjDitherProcessor)
};
