// Compile and run: gcc -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow Drudee.c -lm -o Drude && ./Drude

#include <stdio.h>
#include <math.h>
#include <complex.h>

#define SIZE 200
#define DISPERSION_LAYER 120
#define DISPERSION_LAYER_END 200
#define MAX_TIME 1000

#define PROBE_POSITION 100



struct Data {
    double ez[SIZE];
    double hy[SIZE];
    double jp[SIZE];
    double old_ez[SIZE];
    double cje[SIZE];
    double cjj[SIZE];
    double ca[SIZE];
    double cb[SIZE];

    double saved[MAX_TIME];
};

struct Constants {
    double imp0;
    double e_d; // Debye strength: epsilon_static - epsilon_infinity
    int Nt;
    double sigma;
    double Sc;
    double dt;
    double abc_coef;
    double eps0;
    double eps_inf;
    double Ng;
    double pi;
    double Np; 
};

void initialize_constant(struct Constants *constants) {
    constants->imp0 = 377.0;
    constants->e_d = 4.0;
    constants->Nt = 50;
    constants->sigma = 0.0;
    constants->Sc = 1.0;
    constants->dt = 1.0;
    constants->abc_coef = (constants->Sc - 1.0) / (constants->Sc + 1.0);
    constants->eps0 = 8.854187817e-12;
    constants->eps_inf = 2.0;
    constants->Ng = 50; 
    constants->pi = acos(-1.0); 
    constants->Np = 20;
}


void initialize_grid(const struct Constants *constants, struct Data *data, int dispersive) {
    if(dispersive) {
        double A = (0.5/constants->Ng); 
        double B = (2*pow(constants->pi, 2) * constants->Sc) / (constants->imp0 * pow(constants->Np,2)) ; 

        double L = (constants->sigma * constants->dt) / (2 * constants->eps_inf * constants->eps0);
        double N = (constants->imp0 * constants->Sc)/(constants->eps_inf);

        for( int i=DISPERSION_LAYER; i<DISPERSION_LAYER_END; i++){
            
            
            data->cjj[i] = (1.0 - A)/(1.0 + A);
            data->cje[i] = B / ( 1.0 + A);

            double M = (data->cje[i] * constants->imp0 * constants->Sc) / ( 2.0 * constants->eps_inf);          
            double D = 1.0 + L + M ;

            data->ca[i] = (1.0 - L - M) / D;
            data->cb[i] = N / D ;
        }

        for (int i=0; i<DISPERSION_LAYER; i++) {
            data->cjj[i] = 0.0;
            data->cje[i] = 0.0;
            data->ca[i] = 1.0;
            data->cb[i] = constants->imp0 * constants->Sc;
        }

        for (int i=DISPERSION_LAYER_END; i<SIZE; i++) {
            data->cjj[i] = 0.0;
            data->cje[i] = 0.0;
            data->ca[i] = 1.0;
            data->cb[i] = constants->imp0 * constants->Sc;
        }
    } else {
         for (int i=0; i<SIZE; i++) {
            data->cjj[i] = 0.0;
            data->cje[i] = 0.0;
            data->ca[i] = 1.0;
            data->cb[i] = constants->imp0 * constants->Sc;
        }

    }

}


void run_simulation(struct Data *data, const struct Constants *constants) {
    // 1. update magnetic field
    // 2. store old electric field
    // 3. update electric field
    // 4. update polarization current


     for(int qTime=0; qTime < MAX_TIME; qTime++) {
        //update magnetic field
        for (int mm=0; mm<SIZE-1; mm++) {
            data->hy[mm] = data->hy[mm] + (data->ez[mm+1] - data->ez[mm]) / constants->imp0;
        }

        for (int mm=0; mm<SIZE; mm++) {
            data->old_ez[mm] = data->ez[mm];
        }

        //update electric field
        for (int mm=1; mm<SIZE-1; mm++) {
            data->ez[mm] = data->ca[mm] * data->old_ez[mm]
                + data->cb[mm] * ((data->hy[mm] - data->hy[mm-1]) - 0.5 * (1.0 + data->cjj[mm]) * data->jp[mm]);
        }

        //polarization current update
        for (int mm=0; mm<SIZE; mm++) {
            data->jp[mm] = data->cjj[mm] * data->jp[mm] + data->cje[mm] * (data->ez[mm] + data->old_ez[mm]);
        }

        /* first-order ABC for ez[0], simple ABC for ez[SIZE-1] */
        data->ez[0] = data->old_ez[1] + constants->abc_coef * (data->ez[1] - data->old_ez[0]);
        data->ez[SIZE-1] = data->ez[SIZE-2];

        //source excitation
        data->ez[50] += exp(-(qTime + 0.5 - (-0.5) - 30.) * (qTime + 0.5 - (-0.5) - 30.) / 100.);


        // Save the probe place
        data->saved[qTime] = data->ez[PROBE_POSITION];

    }
}


void dft(const double *data, double *magnitude, int k_bins) {
    const double pi = acos(-1.0);

    for(int k=0; k<k_bins; k++) {
        double real_part = 0.0;
        double imag_part = 0.0;

        for(int n=0; n<MAX_TIME; n++) {
            double angle = -2.0 * pi * k * n / MAX_TIME;
            real_part += data[n] * cos(angle);
            imag_part += data[n] * sin(angle);
        }

        magnitude[k] = sqrt(real_part * real_part + imag_part * imag_part);

    }
}

typedef struct {
    double real;
    double imag;
} Complex;

Complex complex_square_root(double x, double y) {

   Complex result;


   if (x == 0.0 && y == 0.0) {
     result.real = 0.0;
     result.imag = y;
     return result;
   }

   /*  z = x + j y
       sqrt(z) = u + i v
       Οποτε
       z = u^2 - v ^2 + j ( 2uv)
       ετσι x = u^2 - v^2 και
       y = 2uv
     */

     /* Επισης  |z|= sqrt(x^2+y^2)=r
        αλλα και |z|=(u+iv)^2=u^2+v^2
        => r = u^2 + v^2

        u^2 = (r + x) / 2 και v^2 = (r - x) / 2

        ετσι η αρχική ριζα sqrt(x+iy)= sqrt((r+x)/2) + i sqrt((r-x)/2)
      */


      double r = hypot(x, y);
      double t;

      if (x >= 0.0) {
        t = sqrt(0.5 * r + 0.5 * x);

        result.real = t;
        result.imag = y / (2.0 * t);
      } else {
        t = sqrt(0.5 * r - 0.5 * x);
        result.imag = copysign(t, y);
        result.real = y / (2.0 * result.imag);
     }

     return result;
}


int main(){
    struct Data data1 = {0};
    struct Data data2 = {0};

    struct Constants constants;

    int dispersive = 1;
    initialize_constant(&constants);
    initialize_grid(&constants, &data1, dispersive);
    run_simulation(&data1, &constants);

    dispersive = 0;
    initialize_grid(&constants, &data2, dispersive);
    run_simulation(&data2, &constants);

    int gate_full = 180;
    int gate_zero = 230;
    double pi = acos(-1.0);

    for(int k=0; k<MAX_TIME; k++){
      double w;

      if(k < gate_full){
        w = 1.0;
      } else if( k >= gate_zero){
        w = 0.0;
      } else {
        w = 0.5 * (1.0 + cos(pi * (double)(k - gate_full) / (gate_zero - gate_full)));
      }

        data1.saved[k] = w * data1.saved[k];
        data2.saved[k] = w * data2.saved[k];
      }


    // the diff is the E_reflected
    double diff[MAX_TIME];
    for(int i=0; i<MAX_TIME; i++) {
        diff[i] = data1.saved[i] - data2.saved[i];
    }

    int k_bins = MAX_TIME / 2 + 1;

    double reflected_magnitude[k_bins];
    dft(diff, reflected_magnitude, k_bins);

    //diff_magnitude contains FFT(reflected)
    double incident_magnitude[k_bins];
    dft(data2.saved, incident_magnitude, k_bins);

    double peak = 0.0;

    for(int i=0; i<k_bins; i++){
       if( peak < incident_magnitude[i] )
         peak = incident_magnitude[i];
    }

    double threshold = 1e-2 * peak ;


    double gamma_simulated[k_bins];
    for (int k=0; k<k_bins; k++){
       if ( incident_magnitude[k] < threshold){
         gamma_simulated[k] = NAN;
       } else {
         gamma_simulated[k] = reflected_magnitude[k] / incident_magnitude[k];
       }
    }


    /*  e_r = eps_inf + e_d / (1 + jωτ)
        παιρνουμε το complex conjugate και πολλαπλασιαζουμε αριθμητη και παρονομαστη
        e_r = eps_inf + e_d (1 - jωτ) / (1 + (ωτ)^2)
        e_r = eps_inf + e_d / (1 + (ωτ)^2) - j e_d (ωτ) / (1 + (ωτ)^2)
    */

  
    double gamma_analytic[k_bins];
    double omega_p = 2.0 * pi * constants.Sc / ( constants.Np * constants.dt) ;
    double g = 1.0 / (constants.Ng * constants.dt);
    for(int k=1; k<k_bins; k++){
      double omega = 2*pi*k / (MAX_TIME * constants.dt);
      double complex eps_r = constants.eps_inf - pow(omega_p,2) / (pow(omega, 2) - I * omega * g);
      double complex refractive_index = csqrt(eps_r);
      gamma_analytic[k] = cabs((1.0 - refractive_index) / (1.0 + refractive_index));
    }

    int counter = 0 ;
    double delta = 0.0 ;
    for(int k=1; k<k_bins; k++)
    {
       if(!isnan(gamma_simulated[k])){
         delta += fabs(gamma_simulated[k] - gamma_analytic[k]);
         counter++;
       }
     }
     if (counter == 0) {
       fprintf(stderr, "καμια συχνοτητα δεν περασε το theeshold.\n");
       return 1;
     }

     double avg_error = delta / counter;
     printf("Μέσος ορος αποκλισης: %.6e (%d συχνοτητες)\n", avg_error, counter);




    // double E_incident[MAX_TIME / 2 + 1];
    // fftw(free_space_data, E_incident);

    // double E_reflected[MAX_TIME / 2 + 1];
    // fftw(diff, E_reflected);

    //analytical = 1 - sqrt(e_r) / 1 + sqrt(e_r)

    // double e_r  = constants.e_d / (1 + (2 * pi * k) / 1000);
    // double analytical = 1 - sqrt(e_r) / (1 + sqrt(e_r));
    // int var = MAX_TIME / 2 + 1;
    // double omega[var];

    // for(int i=0; i<var; i++) {
    //     omega[i] = 1/var * i;
    // }
    // double analytical[var];
    // for (int i=0; i<omega[var-1]; i++) {
    //     analytical[i] = constants.e_d / (constants.eps_inf + j * omega[i]);
    // }




    return 0;

}
