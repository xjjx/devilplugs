#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <cstdint>
#include <vector>

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
    int  getNumPrograms()    override { return 1; }
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
    //==============================================================================
    // One 24-bit LSB in normalised float domain: 1.0 / 2^23
    static constexpr double kOneLSB = 1.0 / 8388608.0;

    // Scaling factor: top 53 bits of uint64 -> double in [0, 1)
    static constexpr double kU64ToDouble = 1.0 / 9007199254740992.0;

    //==============================================================================
    // xoshiro256** state.
    // Two independent instances, one per channel (L/R).
    // state[0..3] = Left, state[4..7] = Right.
    uint64_t state[8] {};

    //==============================================================================
    // Noise scratch buffers — allocated once in prepareToPlay, reused each block.
    // Avoids any heap allocation in the audio thread.
    std::vector<double> noiseL;
    std::vector<double> noiseR;

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

        // Different starting point guarantees independent L/R streams
        s = seed ^ 0xdeadbeefcafe1234ULL;
        for (int i = 4; i < 8; ++i)
            state[i] = splitmix64 (s);
    }

    //==============================================================================
    static inline uint64_t rotl64 (uint64_t x, int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }

    // xoshiro256** — advance one channel state and return next raw uint64.
    // 'base' points to state[0] for Left, state[4] for Right.
    static inline uint64_t xoshiroNext (uint64_t* base) noexcept
    {
        const uint64_t result = rotl64 (base[1] * 5, 7) * 9;
        const uint64_t t      = base[1] << 17;
        base[2] ^= base[0];
        base[3] ^= base[1];
        base[1] ^= base[2];
        base[0] ^= base[3];
        base[2] ^= t;
        base[3]  = rotl64 (base[3], 45);
        return result;
    }

    // Raw uint64 -> double in [-1, +1)
    static inline double toDouble (uint64_t raw) noexcept
    {
        // Use top 53 bits for full double mantissa precision
        return static_cast<double> (raw >> 11) * kU64ToDouble * 2.0 - 1.0;
    }

    // One TPDF sample for a given channel.
    // Sum of two independent uniform [-1, +1) values, averaged and scaled to 1 LSB.
    // Result is triangular over [-1 LSB, +1 LSB).
    inline double tpdfSample (int ch) noexcept
    {
        uint64_t* base = state + (ch * 4);
        const double a = toDouble (xoshiroNext (base));
        const double b = toDouble (xoshiroNext (base));
        return (a + b) * (0.5 * kOneLSB);
    }

    // Fill dst[0..n) with TPDF noise for a given channel.
    // Isolated as a separate loop so the compiler sees a simple
    // sequential write — the addition loop below is then a plain
    // read+add+write which auto-vectorises cleanly to SSE2/AVX.
    void fillNoise (double* dst, size_t n, int ch) noexcept
    {
        for (size_t i = 0; i < n; ++i)
            dst[i] = tpdfSample (ch);
    }

    //==============================================================================
    template <typename FloatType>
    void applyDither (juce::AudioBuffer<FloatType>& buffer);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjDitherProcessor)
};
