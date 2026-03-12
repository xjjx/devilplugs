//==============================================================================
// TransformerWDF.h
// Drop-in replacement for your existing TransformerWDF struct.
// Integrates NL core resistor (Option 2) with your drive/character params.
//==============================================================================
#pragma once
#include <chowdsp_wdf/chowdsp_wdf.h>
#include <cmath>

namespace wdft = chowdsp::wdft;

static const double resistance = 470000.0;

//==============================================================================
// Nonlinear core resistor — models transformer core saturation.
// Uses Newton-Raphson to solve the implicit WDF equation per sample.
//==============================================================================
struct CoreNonlinearResistor
{
    double R_lin = resistance;  // linear core loss resistance
    double alpha = 0.0;      // saturation amount  — set via setParams()
    double beta  = 0.75;     // saturation hardness — set via setParams()
    double Rp    = resistance;  // port resistance (must match R_lin initially)

    void setParams (double newAlpha, double newBeta)
    {
        alpha = newAlpha;
        beta  = newBeta;
        Rp    = R_lin; // port resistance stays linear
    }

    // i(v) = v/R_lin + alpha * tanh(beta * v)
    double reflected (double a_wave)
    {
        double v = a_wave / 2.0; // initial guess

        for (int iter = 0; iter < 4; ++iter)
        {
            double tanhBV = std::tanh (beta * v);
            double i_v    = v / R_lin + alpha * tanhBV;
            double di_v   = 1.0 / R_lin + alpha * beta * (1.0 - tanhBV * tanhBV);
            double F      = v + Rp * i_v - a_wave;
            double dF     = 1.0 + Rp * di_v;
            v -= F / dF;
        }

        double i_v = v / R_lin + alpha * std::tanh (beta * v);
        return v - Rp * i_v;
    }
};

//==============================================================================
// TransformerWDF — full model with NL core.
//
// Topology:
//   Vs(600Ω) ── S1(series) ── Lleakage
//                                 │
//                            P1(parallel)
//                           /            \
//                       Lm(mag)      coreNL_port
//                           \            /
//                            P2(parallel)
//                                  │
//                               Rload(47kΩ)
//==============================================================================
struct TransformerWDF
{
    //--- Linear WDF elements ------------------------------------------------
    wdft::ResistiveVoltageSourceT<double> Vs       { 600.0   };
    wdft::InductorT<double>               Lleakage { 0.0005  };  // 0.5mH leakage
    wdft::InductorT<double>               Lm       { 2.0     };  // 2H magnetising
    wdft::ResistorT<double>               Rload    { resistance };
    wdft::ResistorT<double>               coreNL_port { resistance }; // matches CoreNL R_lin

    //--- NL core ------------------------------------------------------------
    CoreNonlinearResistor coreNL;

    //--- WDF tree -----------------------------------------------------------
    wdft::WDFSeriesT<double, decltype(Vs), decltype(Lleakage)>    S1   { Vs, Lleakage };
    wdft::WDFParallelT<double, decltype(Lm), decltype(coreNL_port)> P1 { Lm, coreNL_port };
    wdft::WDFParallelT<double, decltype(S1), decltype(P1)>        P2   { S1, P1 };
    wdft::WDFParallelT<double, decltype(P2), decltype(Rload)>     root { P2, Rload };

    //--- DC state -----------------------------------------------------------
    double dcState  = 0.0;
    double dcCoeff  = 0.9999; // recalculated in prepare()

    //------------------------------------------------------------------------
    void prepare (double sampleRate)
    {
        Lleakage.prepare (sampleRate);
        Lm.prepare       (sampleRate);

        // Tune DC blocker pole to ~5Hz regardless of sample rate
        dcCoeff = 1.0 - (2.0 * juce::MathConstants<double>::pi * 5.0 / sampleRate);
    }

    // Called per-block from processImpl with your existing drive/character params
    // driveNorm = 0..1, character = 0..1
    void setDriveParams (double driveNorm, double character)
    {
        // alpha: controls saturation onset
        //   driveNorm^2 gives gentler knee at low drive (matches your existing curve)
        //   character shifts the harmonic flavour: low = even harmonics, high = odd
        double alpha = 0.000015 * (driveNorm * driveNorm) * (1.0 + character * 1.5);

        // beta: softness of the knee
        //   low character = softer (more 2nd harmonic, Neve-like warmth)
        //   high character = harder knee (more 3rd harmonic, more aggressive)
        double beta = 0.5 + character * 0.8;

        coreNL.setParams (alpha, beta);
    }

	double process (double input)
	{
		// 1. DC block ONLY to prevent flux runaway — not for audio shaping
		//    Use a very slow integrator, only tracks true DC offset
		dcState = dcCoeff * dcState + (1.0 - dcCoeff) * input;

		// *** KEY FIX: feed full input to WDF, only use dcState for NL flux tracking ***
		// Don't subtract dcState from the signal path itself
		double fluxEstimate = input - dcState;

		// 2. NL core perturbation uses flux estimate, not the main signal
		double b_nl = coreNL.reflected (fluxEstimate);
		double v_nl = (fluxEstimate + b_nl) / 2.0;

		// 3. Feed full unmodified input + small NL correction to WDF tree
		Vs.setVoltage (input + v_nl * 0.12);

		root.incident (0.0);
		root.reflected();

        static constexpr double gainCompensation = 1.58489319; // 10^(4.0/20) = +4dB
        double out = wdft::voltage<double> (Rload);

        return out * gainCompensation;
	}
};
