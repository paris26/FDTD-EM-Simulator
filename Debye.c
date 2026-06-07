#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>

#define SIZE 200
#define DISPERSION_LAYER 120
#define DISPERSION_LAYER_END 150
#define SNAPSHOT_INTERVAL 10
#define SNAPSHOT_DIR "snapshots"

int main(){
    double ez[SIZE] , hy[SIZE], jp[SIZE], old_ez[SIZE], cje[SIZE], cjj[SIZE], ca[SIZE], cb[SIZE], denom[SIZE];
    int maxTime = 1000, mm;
    double imp0 = 377.0; // Impedance of free space
    double e_d = 4.0; // Relative permittivity of the material
    int Nt = 50; 
    double sigma = 0.0; // Conductivity of the material
    double Sc = 1.0;
    double dt = 1.0;
    double abc_coef = (Sc - 1.0) / (Sc + 1.0);
    double eps0 = 8.854187817e-12; // Permittivity of free space
    double eps_inf = 2.0; // High-frequency permittivity

    char basename[80] = "sim", filename[120];
	int frame = 0;
	FILE *snapshot;

    if (mkdir(SNAPSHOT_DIR, 0777) != 0 && errno != EEXIST) {
        perror(SNAPSHOT_DIR);
        return 1;
    }


    for (int i = 0; i < SIZE; i++) {
        ez[i] = 0.0;
        hy[i] = 0.0;
        jp[i] = 0.0;
        cje[i] = 0.0;
        cjj[i] = 0.0;
        ca[i] = 1.0;
        cb[i] = imp0 * Sc;
        denom[i] = 1.0;
    }

    // 20 spatial steps for the dispersion layer ( Debye ) 
    for (int i=0; i < SIZE; i++) {
        if (i >= DISPERSION_LAYER && i < DISPERSION_LAYER_END) {
            cjj[i] = (1.0 - 0.5 / Nt) / (1.0 + 0.5 / Nt);
            cje[i] = ((e_d - eps_inf) / (imp0 * Sc)) * (1.0 / Nt) / (1.0 + 0.5 / Nt);
        }
    }

    // Ca , cb calculation 
    for (int i=0; i < SIZE; i++) {
        if (i >= DISPERSION_LAYER && i < DISPERSION_LAYER_END) {
            denom[i] =
                        1.0
                        + (sigma * dt) / (2.0 * eps_inf * eps0)
                        + (cje[i] * imp0 * Sc) / (2.0 * eps_inf);

            ca[i] =
                (
                    1.0
                    - (sigma * dt) / (2.0 * eps_inf * eps0)
                    + (cje[i] * imp0 * Sc) / (2.0 * eps_inf)
                ) / denom[i];

            cb[i] =
                    (imp0 * Sc / eps_inf) / denom[i];
        }
    }
    

    for(int qTime=0; qTime < maxTime; qTime++) {
        //update magnetic field
        for (mm=0; mm<SIZE-1; mm++) {
            hy[mm] = hy[mm] + (ez[mm+1] - ez[mm]) / imp0;
        }

        for (mm=0; mm<SIZE; mm++) {
            old_ez[mm] = ez[mm];
        }

        //update electric field
        for (mm=1; mm<SIZE-1; mm++) {
            ez[mm] = ca[mm] * old_ez[mm]
                + cb[mm] * ((hy[mm] - hy[mm-1]) - 0.5 * (1.0 + cjj[mm]) * jp[mm]);
        }

        //polarization current update
        for (mm=0; mm<SIZE; mm++) {
            jp[mm] = cjj[mm] * jp[mm] + cje[mm] * (ez[mm] - old_ez[mm]);
        }

        /* first-order ABC for ez[0], simple ABC for ez[SIZE-1] */
        ez[0] = old_ez[1] + abc_coef * (ez[1] - old_ez[0]);
        ez[SIZE-1] = ez[SIZE-2];

        //source excitation
        ez[50] += exp(-(qTime + 0.5 - (-0.5) - 30.) * (qTime + 0.5 - (-0.5) - 30.) / 100.);

        /* write snapshot if time a multiple of SNAPSHOT_INTERVAL */
        if (qTime % SNAPSHOT_INTERVAL == 0) {
            snprintf(filename, sizeof(filename), "%s/%s.%d", SNAPSHOT_DIR, basename, frame++);
            snapshot=fopen(filename, "w");
            if (snapshot == NULL) {
                perror(filename);
                return 1;
            }
            for (mm = 0; mm < SIZE; mm++)
            fprintf(snapshot, "%g\n", ez[mm]);
            fclose(snapshot);
        }
    }

    return 0;

}
