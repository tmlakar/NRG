using System;
using MathNet.Numerics.Integration;

namespace PathTracer
{
    /// <summary>
    /// Sphere Shape template class - NOT implemented completely
    /// </summary>
    class Sphere : Shape
    {
        public double Radius { get; set; }
        public Sphere(double radius, Transform objectToWorld)
        {
            Radius = radius;
            ObjectToWorld = objectToWorld;
        }

        /// <summary>
        /// Ray-Sphere intersection - NOT implemented
        /// </summary>
        /// <param name="r">Ray</param>
        /// <returns>t or null if no hit, point on surface</returns>
        public override (double?, SurfaceInteraction) Intersect(Ray ray)
        {
            // https://www.pbr-book.org/4ed/Shapes/Spheres#IntersectionTests
            // transform ray to object space
            Ray r = WorldToObject.Apply(ray);

            // TODO: Compute quadratic sphere coefficients
            // TODO: Initialize _double_ ray coordinate values
            // origin coordinates
            double ox = r.o.x;
            double oy = r.o.y;
            double oz = r.o.z;
            
            // direction coordinates
            double dx = r.d.x;
            double dy = r.d.y;
            double dz = r.d.z;
            
            //coefficients
            double a = dx * dx + dy * dy + dz * dz;
            double b = 2 * (dx * ox + dy * oy + dz * oz);
            double c = ox * ox + oy * oy + oz * oz - (this.Radius * this.Radius);

            
            // TODO: Solve quadratic equation for _t_ values
            // at^2 + bt + c = 0
            (bool equation, double t0, double t1) = Utils.Quadratic(a, b, c);
            
            // TODO: Check quadric shape _t0_ and _t1_ for nearest intersection
            if (!equation) {
                // no real solution
                return (null, null);
            }
            
            // Utils.Quadratic always returns t0 as the smaller value between the two
            // no need to check for tMax since the ray is infinite
            double t = t0;
            if (t0 <= 0 && t1 <= 0)
            {
                return (null, null);
            }
            
            if (t0 <= 0 && t1 > 0)
            {
                t = t1;
            }

            // TODO: Compute sphere hit position and $\phi$
            // Ray.Point returns point on ray given parameter t
            Vector3 tHitPosition = r.Point(t);
            
            // refine sphere intersection point
            if (tHitPosition.x == 0 && tHitPosition.y == 0)
                tHitPosition.x = 1e-5f * Radius;
            
            
            double phi = Math.Atan2(tHitPosition.y, tHitPosition.x);
            if (phi < 0)
            {
                phi = phi + 2 * Math.PI;
            }
            
            
            // TODO: Return shape hit and surface interaction
            Vector3 wo = -r.d;
            Vector3 normal = tHitPosition.Clone().Normalize();
            Vector3 dpdu = new Vector3(-tHitPosition.y, tHitPosition.x, 0);
                
            SurfaceInteraction surfaceInteraction = new SurfaceInteraction(tHitPosition, normal, wo, dpdu, this);
            return (t, ObjectToWorld.Apply(surfaceInteraction));

            // A dummy return example
            // double dummyHit = 0.0;
            // Vector3 dummyVector = new Vector3(0, 0, 0);
            // SurfaceInteraction dummySurfaceInteraction = new SurfaceInteraction(dummyVector, dummyVector, dummyVector, dummyVector, this);
            // return (dummyHit, dummySurfaceInteraction);
        }

        /// <summary>
        /// Sample point on sphere in world
        /// </summary>
        /// <returns>point in world, pdf of point</returns>
        public override (SurfaceInteraction, double) Sample()
        {
            // TODO: Implement Sphere sampling
            Vector3 sampledPoint = Samplers.UniformSampleSphere();
            sampledPoint = sampledPoint * this.Radius;
            sampledPoint = ObjectToWorld.ApplyVector(sampledPoint);
            
            // TODO: Return surface interaction and pdf
            Vector3 normal = new Vector3(sampledPoint.x, sampledPoint.y, sampledPoint.z).Normalize();
            Vector3 dpdu = new Vector3(-sampledPoint.y, sampledPoint.x, 0);
            double pdf = 1 / Area();
            SurfaceInteraction surfaceI = new SurfaceInteraction(sampledPoint, normal, Vector3.ZeroVector, dpdu, this);

            return (ObjectToWorld.Apply(surfaceI), pdf);

            // A dummy return example
            // double dummyPdf = 1.0;
            // Vector3 dummyVector = new Vector3(0, 0, 0);
            // SurfaceInteraction dummySurfaceInteraction = new SurfaceInteraction(dummyVector, dummyVector, dummyVector, dummyVector, this);
            // return (dummySurfaceInteraction, dummyPdf);
        }

        public override double Area() { return 4 * Math.PI * Radius * Radius; }

        /// <summary>
        /// Estimates pdf of wi starting from point si
        /// </summary>
        /// <param name="si">point on surface that wi starts from</param>
        /// <param name="wi">wi</param>
        /// <returns>pdf of wi given this shape</returns>
        public override double Pdf(SurfaceInteraction si, Vector3 wi)
        {
            throw new NotImplementedException();
        }

    }
}
