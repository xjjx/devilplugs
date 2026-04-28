#pragma once

#include <emmintrin.h>
#include <juce_audio_processors/juce_audio_processors.h>

class XjDitherProcessor : public juce::AudioProcessor
{
public:
    XjDitherProcessor();
    ~XjDitherProcessor() override = default;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock (juce::AudioBuffer<float>&,  juce::MidiBuffer&) override;
    void processBlock (juce::AudioBuffer<double>&, juce::MidiBuffer&) override;

    //==============================================================================
    // No editor — nothing to control on a flat TPDF dither.
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

    // Scaling factor to convert a raw uint64 to double in [0, 1):
    // uses the top 53 bits  →  1.0 / 2^53
    static constexpr double kU64ToDouble = 1.0 / 9007199254740992.0;

    //==============================================================================
    // xoshiro256** state — 4× uint64 per logical RNG instance.
    // We keep two instances (one per stereo channel) laid out so that
    // the SSE2 path can operate on both simultaneously.
    //
    // Layout:
    //   state[0..3]  → channel 0 (Left)
    //   state[4..7]  → channel 1 (Right)
    //
    // In the SSE2 path we load state[i] and state[i+4] into a __m128i
    // and advance both channels in parallel.
    //
    // Aligned to 16 bytes so SSE2 loads are always aligned.
    alignas(16) uint64_t state[8] {};

    //==============================================================================
    // splitmix64 — used only during seeding, not in the audio path.
    static uint64_t splitmix64 (uint64_t& x) noexcept
    {
        x += 0x9e3779b97f4a7c15ULL;
        x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    void seedRNG (uint64_t seed) noexcept
    {
        // Seed Left channel (indices 0-3)
        uint64_t s = seed;
        for (int i = 0; i < 4; ++i)
            state[i] = splitmix64 (s);

        // Seed Right channel (indices 4-7) from a different starting point
        // so the two streams are guaranteed independent.
        s = seed ^ 0xdeadbeefcafe1234ULL;
        for (int i = 4; i < 8; ++i)
            state[i] = splitmix64 (s);
    }

    //==============================================================================
    // Scalar xoshiro256** step for a single channel.
    // 'base' points to the start of a 4-element state block.
    static inline uint64_t xoshiroNext (uint64_t* base) noexcept
    {
        const uint64_t result = rotl (base[1] * 5, 7) * 9;
        const uint64_t t      = base[1] << 17;
        base[2] ^= base[0];
        base[3] ^= base[1];
        base[1] ^= base[2];
        base[0] ^= base[3];
        base[2] ^= t;
        base[3]  = rotl (base[3], 45);
        return result;
    }

    static inline uint64_t rotl (uint64_t x, int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }

    // Scalar: one TPDF sample for a given channel (0 = L, 1 = R).
    // Returns value in [-1 LSB, +1 LSB).
    inline double tpdfScalar (int ch) noexcept
    {
        uint64_t* base = state + (ch * 4);
        const double a = static_cast<double> (xoshiroNext (base) >> 11) * kU64ToDouble * 2.0 - 1.0;
        const double b = static_cast<double> (xoshiroNext (base) >> 11) * kU64ToDouble * 2.0 - 1.0;
        return (a + b) * 0.5 * kOneLSB;
    }

    //==============================================================================
    // SSE2 path: advance both channels simultaneously and return two TPDF
    // samples packed into a __m128d  { noiseL, noiseR }.
    //
    // xoshiro256** in SSE2:
    //   We store Left state in state[0..3] and Right state in state[4..7].
    //   Each __m128i holds { state[i], state[i+4] } — one element per channel.
    //   All four 128-bit words are updated in parallel across both channels.
    //
    // This halves the number of scalar instructions compared to two separate
    // xoshiro calls, at the cost of slightly more complex index bookkeeping.
    inline __m128d tpdfSSE2() noexcept
    {
        // Load state words interleaved: lo = Left[i], hi = Right[i]
        __m128i s0 = _mm_set_epi64x ((int64_t)state[4], (int64_t)state[0]);
        __m128i s1 = _mm_set_epi64x ((int64_t)state[5], (int64_t)state[1]);
        __m128i s2 = _mm_set_epi64x ((int64_t)state[6], (int64_t)state[2]);
        __m128i s3 = _mm_set_epi64x ((int64_t)state[7], (int64_t)state[3]);

        // result = rotl(s1 * 5, 7) * 9
        // SSE2 has no 64-bit multiply, so we use the 32-bit multiply trick:
        // For 64-bit a*b we only need the low 64 bits, which equals
        // (a_lo * b_lo) + ((a_lo * b_hi + a_hi * b_lo) << 32).
        // Since 5 and 9 are small constants we can use _mm_mul_epu32 directly
        // on both halves and combine.
        auto mul64 = [&](__m128i a, uint64_t scalar) -> __m128i
        {
            // Split scalar into lo/hi 32-bit halves
            const uint32_t lo32 = (uint32_t)(scalar & 0xffffffff);
            const uint32_t hi32 = (uint32_t)(scalar >> 32);

            // lo part: a_lo * scalar_lo  (fits in 64 bits via _mm_mul_epu32)
            __m128i prod_lo = _mm_mul_epu32 (a, _mm_set1_epi32 ((int)lo32));

            // hi part: a_lo * scalar_hi, shifted left 32
            __m128i a_lo    = _mm_and_si128 (a, _mm_set1_epi32 (0xffffffff));
            __m128i prod_hi = _mm_mul_epu32 (a_lo, _mm_set1_epi32 ((int)hi32));
            prod_hi         = _mm_slli_epi64 (prod_hi, 32);

            // a_hi * scalar_lo, shifted left 32
            __m128i a_hi     = _mm_srli_epi64 (a, 32);
            __m128i prod_hi2 = _mm_mul_epu32 (a_hi, _mm_set1_epi32 ((int)lo32));
            prod_hi2         = _mm_slli_epi64 (prod_hi2, 32);

            return _mm_add_epi64 (_mm_add_epi64 (prod_lo, prod_hi), prod_hi2);
        };

        auto rotl64 = [&](__m128i x, int k) -> __m128i
        {
            return _mm_or_si128 (_mm_slli_epi64 (x, k),
                                 _mm_srli_epi64 (x, 64 - k));
        };

        // First output — used for noise draws 'a'
        __m128i resultA = mul64 (rotl64 (mul64 (s1, 5), 7), 9);

        // Advance state (xoshiro256** update)
        auto advance = [&](__m128i& _s0, __m128i& _s1, __m128i& _s2, __m128i& _s3)
        {
            __m128i t = _mm_slli_epi64 (_s1, 17);
            _s2 = _mm_xor_si128 (_s2, _s0);
            _s3 = _mm_xor_si128 (_s3, _s1);
            _s1 = _mm_xor_si128 (_s1, _s2);
            _s0 = _mm_xor_si128 (_s0, _s3);
            _s2 = _mm_xor_si128 (_s2, t);
            _s3 = rotl64 (_s3, 45);
        };

        advance (s0, s1, s2, s3);

        // Second output — used for noise draws 'b'
        __m128i resultB = mul64 (rotl64 (mul64 (s1, 5), 7), 9);

        advance (s0, s1, s2, s3);

        // Write state back
        alignas(16) uint64_t tmp[2];

        _mm_store_si128 ((__m128i*)tmp, s0);
        state[0] = tmp[0]; state[4] = tmp[1];
        _mm_store_si128 ((__m128i*)tmp, s1);
        state[1] = tmp[0]; state[5] = tmp[1];
        _mm_store_si128 ((__m128i*)tmp, s2);
        state[2] = tmp[0]; state[6] = tmp[1];
        _mm_store_si128 ((__m128i*)tmp, s3);
        state[3] = tmp[0]; state[7] = tmp[1];

        // Convert to double in [0, 1) using top 53 bits, then map to [-1, +1)
        // SSE2 has no direct uint64→double, so we use the int64 path with bias:
        // reinterpret bits as double in [1.0, 2.0) and subtract 1.0
        const __m128i mantissaMask = _mm_set1_epi64x (0x000fffffffffffffLL);
        const __m128i exponentOne  = _mm_set1_epi64x (0x3ff0000000000000LL);
        const __m128d one          = _mm_set1_pd (1.0);
        const __m128d two          = _mm_set1_pd (2.0);

        // a in [-1, +1)
        __m128i bitsA = _mm_or_si128 (_mm_and_si128 (resultA, mantissaMask), exponentOne);
        __m128d dA    = _mm_sub_pd (_mm_castsi128_pd (bitsA), one); // [0, 1)
        dA            = _mm_sub_pd (_mm_mul_pd (dA, two), one);     // [-1, +1)

        // b in [-1, +1)
        __m128i bitsB = _mm_or_si128 (_mm_and_si128 (resultB, mantissaMask), exponentOne);
        __m128d dB    = _mm_sub_pd (_mm_castsi128_pd (bitsB), one);
        dB            = _mm_sub_pd (_mm_mul_pd (dB, two), one);

        // TPDF: (a + b) * 0.5 * kOneLSB  — both channels in one instruction set
        const __m128d scale = _mm_set1_pd (0.5 * kOneLSB);
        return _mm_mul_pd (_mm_add_pd (dA, dB), scale);
    }

    //==============================================================================
    template <typename FloatType>
    void applyDither (juce::AudioBuffer<FloatType>& buffer);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (XjDitherProcessor)
};
