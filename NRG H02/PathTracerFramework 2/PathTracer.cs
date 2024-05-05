using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static PathTracer.Samplers;

namespace PathTracer
{
    class PathTracer
    {
        /// <summary>
        /// Given Ray r and Scene s, trace the ray over the scene and return the estimated radiance
        /// </summary>
        /// <param name="r">Ray direction</param>
        /// <param name="s">Scene to trace</param>
        /// <returns>Estimated radiance in the ray direction</returns>
        public Spectrum Li(Ray r, Scene s)
        {
            // L <- 0
            var L = Spectrum.ZeroSpectrum;
            
            // β <- 1
            Spectrum beta = Spectrum.Create(1); 

            // nbounces <- 0
            var nbounces = 0;

            // main loop for tracing
            while(nbounces < 20) {

                // compute intersection point of ray r with scene s
                var (d, isect) = s.Intersect(r); //intersect r with scene

                // if isect == null -> no intersection found, break
                if (isect == null) { 
                    break;
                }
                
                Vector3 wo = -r.d;

                // check if the intersection is with light source
                // if isect == light -> multiply with light emission and break
                if (isect.Obj is Light) {
                    if (nbounces == 0) {
                        L = beta * isect.Le(wo);
                    }
                    break;
                }
                
                // path reusing: sampling light at intersection 
                Spectrum Ldirect = Light.UniformSampleOneLight(isect, s);
                L.AddTo(beta * Ldirect);
                
                // wi <- random ray from isect (sample random ray to get new direction)
                (Spectrum f, Vector3 wi, double pdf, bool isSpecular) = (isect.Obj as Shape).BSDF.Sample_f(wo, isect);
                double cosTheta = Vector3.AbsDot(wi, isect.Normal);
                //  beta is updated based on the sampled direction and BSDF
                beta = beta * f * cosTheta / pdf;
                
                // ray is spawned in the sampled direction for the next iteration
                r = isect.SpawnRay(wi);
                
                // terminate the ray tracing process based on the probability of a ray contribution
                if (nbounces > 3) {
                    double q = 1 - beta.Max();
                  
                    if (ThreadSafeRandom.NextDouble() < q) {
                        break;
                    }
                    
                    beta = beta * 1 / (1 - q);
                }
                // increment number of bounces
                nbounces++;

            }
            return L;
        }

    }
}
