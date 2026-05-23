#include "PluginProcessor.h"

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
InputTransformerAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterChoice>(
        "mode", "Mode",
        juce::StringArray { "A", "S", "N" }, 0));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "drive", "Drive",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f, 0.5f), 0.5f));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "trim", "Trim",
        juce::NormalisableRange<float>(-6.0f, 6.0f, 0.1f), 0.0f));

    return layout;
}

//==============================================================================
InputTransformerAudioProcessor::InputTransformerAudioProcessor()
    : AudioProcessor(BusesProperties()
        .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
}

//==============================================================================
void InputTransformerAudioProcessor::prepareToPlay(double sr, int)
{
    sampleRate = sr;
    const double pi2 = juce::MathConstants<double>::twoPi;

    // ── Pre/de-emphasis nominal shelf coefficients ────────────────────────────
    // 1-pole LP coeff — higher freq = less LP = more HF emphasis
    coeffs.a_pre = std::exp(-pi2 * 18000.0 / sr);  // Mode A: gentle ~18kHz
    coeffs.s_pre = std::exp(-pi2 * 15000.0 / sr);  // Mode S: ~15kHz
    coeffs.n_pre = std::exp(-pi2 * 10000.0 / sr);  // Mode N: deeper ~10kHz

    // ── DC blocker ~5Hz ───────────────────────────────────────────────────────
    coeffs.dc = std::exp(-pi2 * 5.0 / sr);

    // ── Mode N: 2-pole resonant LF bump ~60Hz, Q=1.8, +3dB ──────────────────
    {
        const double f0    = 60.0;
        const double Q     = 1.8;
        const double A     = std::pow(10.0, 3.0 / 40.0);
        const double w0    = pi2 * f0 / sr;
        const double cos0  = std::cos(w0);
        const double alpha = std::sin(w0) / (2.0 * Q);
        const double a0    =  1.0 + alpha / A;

        coeffs.n_lf_b0 = (1.0 + alpha * A) / a0;
        coeffs.n_lf_b1 = (-2.0 * cos0)     / a0;
        coeffs.n_lf_b2 = (1.0 - alpha * A) / a0;
        coeffs.n_lf_a1 = (-2.0 * cos0)     / a0;
        coeffs.n_lf_a2 = (1.0 - alpha / A) / a0;
    }

    // ── Hysteresis allpass coefficients ───────────────────────────────────
    // Base coeff sets nominal phase frequency; depth scales with signal level.
    // 1-pole allpass: coeff near 0 = phase shift ~fs/2; coeff near 1 = near DC.
    // We target phase smearing around 2–4kHz at rest, drifting lower on peaks.
    coeffs.a_apBase  = std::exp(-pi2 * 3000.0 / sr);  // Mode A: subtle
    coeffs.a_apDepth = 0.06;                           // small level-dependent shift

    coeffs.s_apBase  = std::exp(-pi2 * 2500.0 / sr);  // Mode S: moderate
    coeffs.s_apDepth = 0.09;

    coeffs.n_apBase  = std::exp(-pi2 * 1800.0 / sr);  // Mode N: strongest
    coeffs.n_apDepth = 0.14;

    // ── Thermal drift ─────────────────────────────────────────────────────
    // Random walk runs at audio rate but is heavily smoothed (~2–5s time constant)
    // Step size is tiny — the walk accumulates over thousands of samples
    coeffs.driftStep   = 0.0003;                       // per-sample walk increment
    coeffs.driftSmooth = std::exp(-1.0 / (sr * 3.0)); // ~3s smoothing
    coeffs.driftRange  = 0.08;                         // ±8% frequency deviation

    // ── Core IM release coefficients ──────────────────────────────────────
    coeffs.a_coreRls = std::exp(-1.0 / (sr * 0.080)); // Mode A: ~80ms
    coeffs.s_coreRls = std::exp(-1.0 / (sr * 0.120)); // Mode S: ~120ms
    coeffs.n_coreRls = std::exp(-1.0 / (sr * 0.180)); // Mode N: ~180ms

    // ── Reset all state ───────────────────────────────────────────────────────
    modeA      = ModeState{};
    modeS      = ModeState{};
    modeN      = ModeState{};
    modeNExtra = ModeNExtra{};
    driftA     = ThermalDrift{};
    driftS     = ThermalDrift{};
    driftN     = ThermalDrift{};
    _lcg       = 0x12345678u;
}

//==============================================================================
void InputTransformerAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                   juce::MidiBuffer&)
{
    processImpl(buffer);
}

void InputTransformerAudioProcessor::processBlock(juce::AudioBuffer<double>& buffer,
                                                   juce::MidiBuffer&)
{
    processImpl(buffer);
}

//==============================================================================
template <typename Sample>
void InputTransformerAudioProcessor::processImpl(juce::AudioBuffer<Sample>& buffer)
{
    juce::ScopedNoDenormals noDenormals;

    const int  numSamples  = buffer.getNumSamples();
    const int  numChannels = buffer.getNumChannels();
    const bool mono        = (numChannels == 1);

    Sample* inL = buffer.getWritePointer(0);
    Sample* inR = mono ? nullptr : buffer.getWritePointer(1);

    const int   mode       = static_cast<int>(*apvts.getRawParameterValue("mode"));
    const float driveParam = *apvts.getRawParameterValue("drive");
    const float trimParam  = *apvts.getRawParameterValue("trim");

    // Drive 0..2 -> internal sat drive 1..6 (exponential feel)
    const double drive    = 1.0 + std::pow((double)driveParam / 2.0, 1.5) * 5.0;
    const double emphBase = (double)driveParam / 2.0;
    const double trim     = std::pow(10.0, (double)trimParam / 20.0);
    const double wet      = juce::jlimit(0.0, 1.0, (double)driveParam);
    const double dry      = 1.0 - wet;

    for (int n = 0; n < numSamples; ++n)
    {
        const double L = static_cast<double>(*inL);
        const double R = mono ? 0.0 : static_cast<double>(*inR);
        double outL = L, outR = R;

        // One LCG call per sample — shared across all movement effects
        const double rnd = lcgRand();

        switch (mode)
        {
            // ── Mode A: even harmonics, gentle ───────────────────────────────
            case 0:
            {
                const double emph = emphBase * 0.4;

                // Thermal drift — wander the pre-emphasis frequency slowly
                const double dA   = thermalStep(driftA, coeffs.driftStep,
                                                coeffs.driftSmooth, rnd);
                const double preC = driftedCoeff(coeffs.a_pre, dA,
                                                 coeffs.driftRange, sampleRate);

                // Hysteresis — allpass phase smear, depth scaled by envelope
                const double apC = coeffs.a_apBase;
                double fL = allpass1(L, modeA.apL, apC);
                double fR = allpass1(R, modeA.apR, apC);

                // Pre-emphasis (drifted coeff)
                fL = preEmphasis(fL, modeA.preL, preC, emph);
                fR = preEmphasis(fR, modeA.preR, preC, emph);

                // Core IM — envelope shifts drive point
                const double eL    = coreEnvelope(fL, modeA.envL, coeffs.a_coreRls);
                const double eR    = coreEnvelope(fR, modeA.envR, coeffs.a_coreRls);
                const double eMono = (eL + eR) * 0.5;
                const double eDrive = drive + eMono * 0.25; // sensitivity A

                // Saturation
                fL = satModeA(fL, eDrive);
                fR = satModeA(fR, eDrive);

                // De-emphasis (same drifted coeff — cancels shelf exactly)
                fL = deEmphasis(fL, modeA.preL, emph);
                fR = deEmphasis(fR, modeA.preR, emph);

                outL = dcBlock(fL, modeA.dcL, modeA.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeA.dcR, modeA.dcHpR, coeffs.dc);
                break;
            }

            // ── Mode S: odd harmonics, punchy ────────────────────────────────
            case 1:
            {
                const double emph = emphBase * 0.55;

                const double dS   = thermalStep(driftS, coeffs.driftStep,
                                                coeffs.driftSmooth, rnd);
                const double preC = driftedCoeff(coeffs.s_pre, dS,
                                                 coeffs.driftRange, sampleRate);

                const double apC = coeffs.s_apBase;
                double fL = allpass1(L, modeS.apL, apC);
                double fR = allpass1(R, modeS.apR, apC);

                fL = preEmphasis(fL, modeS.preL, preC, emph);
                fR = preEmphasis(fR, modeS.preR, preC, emph);

                const double eL    = coreEnvelope(fL, modeS.envL, coeffs.s_coreRls);
                const double eR    = coreEnvelope(fR, modeS.envR, coeffs.s_coreRls);
                const double eMono = (eL + eR) * 0.5;
                const double eDrive = drive + eMono * 0.35; // sensitivity S

                fL = satModeS(fL, eDrive);
                fR = satModeS(fR, eDrive);

                fL = deEmphasis(fL, modeS.preL, emph);
                fR = deEmphasis(fR, modeS.preR, emph);

                outL = dcBlock(fL, modeS.dcL, modeS.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeS.dcR, modeS.dcHpR, coeffs.dc);
                break;
            }

            // ── Mode N: aggressive even harmonics + LF resonance ─────────────
            case 2:
            {
                const double emph = emphBase * 0.7;

                // Thermal drift on pre-emphasis freq
                const double dN   = thermalStep(driftN, coeffs.driftStep,
                                                coeffs.driftSmooth, rnd);
                const double preC = driftedCoeff(coeffs.n_pre, dN,
                                                 coeffs.driftRange, sampleRate);

                // LF resonant bump — static (its own drift omitted for now)
                double fL = biquadTDF2(L, modeNExtra.lf1L, modeNExtra.lf2L,
                                       coeffs.n_lf_b0, coeffs.n_lf_b1, coeffs.n_lf_b2,
                                       coeffs.n_lf_a1, coeffs.n_lf_a2);
                double fR = biquadTDF2(R, modeNExtra.lf1R, modeNExtra.lf2R,
                                       coeffs.n_lf_b0, coeffs.n_lf_b1, coeffs.n_lf_b2,
                                       coeffs.n_lf_a1, coeffs.n_lf_a2);

                const double apC = coeffs.n_apBase;
                fL = allpass1(fL, modeN.apL, apC);
                fR = allpass1(fR, modeN.apR, apC);

                fL = preEmphasis(fL, modeN.preL, preC, emph);
                fR = preEmphasis(fR, modeN.preR, preC, emph);

                const double eL    = coreEnvelope(fL, modeN.envL, coeffs.n_coreRls);
                const double eR    = coreEnvelope(fR, modeN.envR, coeffs.n_coreRls);
                const double eMono = (eL + eR) * 0.5;
                const double eDrive = drive + eMono * 0.45; // sensitivity N

                fL = satModeN(fL, eDrive);
                fR = satModeN(fR, eDrive);

                fL = deEmphasis(fL, modeN.preL, emph);
                fR = deEmphasis(fR, modeN.preR, emph);

                outL = dcBlock(fL, modeN.dcL, modeN.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeN.dcR, modeN.dcHpR, coeffs.dc);
                break;
            }

            default:
                outL = L;
                outR = R;
                break;
        }

        *inL++ = static_cast<Sample>((dry * L + wet * outL) * trim);
        if (!mono)
            *inR++ = static_cast<Sample>((dry * R + wet * outR) * trim);
    }
}

//==============================================================================
juce::AudioProcessorEditor* InputTransformerAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void InputTransformerAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void InputTransformerAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new InputTransformerAudioProcessor();
}
