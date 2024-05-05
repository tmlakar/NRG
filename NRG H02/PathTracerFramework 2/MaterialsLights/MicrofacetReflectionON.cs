using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace PathTracer
{
    /// <summary>
    /// Implementation of Oren-Nayar Microfacet surface
    /// </summary>
    public class MicrofacetReflectionON : BxDF
    {
        // albedo
        private Spectrum kd;
        
        // roughness parameter
        private double sigma;
        
        
        public MicrofacetReflectionON(Spectrum r, double roughness)
        {
            kd = r;
            sigma = roughness;
        }

        /// <summary>
        /// Oren-Nayar f
        /// </summary>
        /// <param name="wo">output vector</param>
        /// <param name="wi">input vector</param>
        /// <returns></returns>
        public override Spectrum f(Vector3 wo, Vector3 wi)
        {
            if (!Utils.SameHemisphere(wo, wi))
            {
                return Spectrum.ZeroSpectrum;
            }

            double a = 1 - (sigma * sigma) / (2 * (sigma * sigma + 0.33));
            double b = 0.45 * (sigma * sigma) / (sigma * sigma + 0.09);
            
            double sinThetaI = Utils.SinTheta(wi);
            double sinThetaO = Utils.SinTheta(wo);

            // max(0, cos(phiI - phi0) => cos(phiI-phi0) = cosphiI*cosphi0 + sinphiIsinphi0
            double maxCos = 0;
            
            double sinPhiI, sinPhiO, cosPhiI, cosPhiO;
            sinPhiI = Utils.SinPhi(wi);
            cosPhiI = Utils.CosPhi(wi);
            sinPhiO = Utils.SinPhi(wo);
            cosPhiO = Utils.CosPhi(wo);
            
            double dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = Math.Max(0.0, dCos);
            
            
            double sinAlpha = 0;
            double tanBeta = 0;
            // check values of cosTheta for wi and wo and set sinAlpha and sinBeta appropriately
            if (Utils.AbsCosTheta(wi) > Utils.AbsCosTheta(wo))
            {
                sinAlpha = sinThetaO;
                tanBeta = sinThetaI / Utils.AbsCosTheta(wi);
            }
            else
            {
                sinAlpha = sinThetaI;
                tanBeta = sinThetaO / Utils.AbsCosTheta(wo);
            }

            
            
            Spectrum f = kd / Math.PI * (a + b * maxCos * sinAlpha * tanBeta);
            return f;
        }

        /// <summary>
        /// Cosine weighted sampling of wi
        /// </summary>
        /// <param name="wo">wo in local</param>
        /// <returns>(f, wi, pdf)</returns>
        public override (Spectrum, Vector3, double) Sample_f(Vector3 wo)
        {
            var wi = Samplers.CosineSampleHemisphere();
            if (wo.z < 0)
                wi.z *= -1;
            double pdf = Pdf(wo, wi);
            return (f(wo, wi), wi, pdf);
        }

        /// <summary>
        /// returns pdf(wo,wi) as |cosTheta|/pi
        /// </summary>
        /// <param name="wo">output vector in local</param>
        /// <param name="wi">input vector in local</param>
        /// <returns></returns>
        public override double Pdf(Vector3 wo, Vector3 wi)
        {
            if (!Utils.SameHemisphere(wo, wi))
                return 0;

            return Math.Abs(wi.z) * Utils.PiInv; // wi.z == cosTheta
        }
    }
}