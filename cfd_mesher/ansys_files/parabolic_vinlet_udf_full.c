/*
vprofile_transient_parabolic.c
UDF for specifying steady-state velocity profile boundary condition

AUTHOR	Wenhao Ding
DATE	22.03.2024

Instruction:
define VINREF, the maximum velocity at the center
define FIXED_TIME_STEP and make sure MAX_PCD_SIZE is larger than int(total_time/FIXED_TIME_STEP)
define filename (txt for the waveform, this txt file must be in the same directory as this script)
*/

#include "udf.h"
#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846
#define VINREF	1.3158	/* ref inlet v  */
#define VIN	0	/* ref inlet v  */
#define MAX_PCD_SIZE 400001  /* Least PCD number the waveform file has cannot exceed point number in txt 574578 41160*/
#define MAX_LOG_SIZE 600
#define FIXED_TIME_STEP 0.00001  /* Fixed time step during transient simulation*/
#define MAX_POINTS 6000 /* Maximum point number on the inlet*/
#define ND_ND 3
/* static double* Q_array[MAX_PCD_SIZE]; */


typedef struct {
    double x;
    double y;
    double z;
} Vector3D;

double dot_product(Vector3D v1, Vector3D v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector3D cross_product(Vector3D v1, Vector3D v2) {
    Vector3D cross_v;

    cross_v.x = v1.y * v2.z - v1.z * v2.y;
    cross_v.y = v1.z * v2.x - v1.x * v2.z;
    cross_v.z = v1.x * v2.y - v1.y * v2.x;

    return cross_v;
}

double get_norm(Vector3D result) {
    return sqrt(pow(result.x, 2) + pow(result.y, 2) + pow(result.z, 2));
}

double getDegAngle(Vector3D v1, Vector3D v2) {
    Vector3D v_cross = cross_product(v1, v2);
    double radian_angle = atan2(get_norm(v_cross), dot_product(v1, v2));
    if (v_cross.z < 0) {
        radian_angle = 2 * PI - radian_angle;
    }
    return radian_angle;
}



DEFINE_PROFILE(inlet_x_velocity_transient, thread, position)
{   
    /*Domain* d;*/
    /*d = Get_Domain(1);*/
    /*Thread* tr = Lookup_Thread(d, 5);*/
    real x[ND_ND]; /* ND_ND=3 for 3D */
    real curr_x, curr_y, curr_z;
    face_t f;
    /* real current_time;  */
    FILE* rfile; /* declare a FILE pointer */
    rfile = fopen("udf_waveform_5cycles.txt", "r+");
    double Q_array[MAX_PCD_SIZE];
    int i = 0;
    for (i = 0; i < MAX_PCD_SIZE; i++) {
        fscanf(rfile, "%lf", &Q_array[i]); /* that supposes one value per line */
    }fclose(rfile);
    /* load from csv file */
    double centroids[MAX_POINTS][ND_ND];
    int num_points_inlet = 0;
    FILE* file;
    char line[256];
    file = fopen("inlet_centroids.csv", "r");
    if (file == NULL)
    {
        Message("Error: Cannot open file %s\n", "inlet_centroids.csv");
        return;
    }
    fgets(line, sizeof(line), file);
    num_points_inlet = 0;
    while (fgets(line, sizeof(line), file) && num_points_inlet < MAX_POINTS)
    {
        sscanf(line, "%lf,%lf,%lf", &centroids[num_points_inlet][0], &centroids[num_points_inlet][1], &centroids[num_points_inlet][2]);
        num_points_inlet++;
    }
    fclose(file);
    /* record average coordinates on the inlet (center) */
    double center_x = 0.0, center_y = 0.0, center_z = 0.0;
    for (i = 0; i < num_points_inlet; i++)
    {
        center_x += centroids[i][0];
        center_y += centroids[i][1];
        center_z += centroids[i][2];
    }

    if (num_points_inlet > 0)
    {
        center_x /= num_points_inlet;
        center_y /= num_points_inlet;
        center_z /= num_points_inlet;
    }
    /*Message("Average centroid coordinates: x=%lf, y=%lf, z=%lf\n", center_x, center_y, center_z);*/
    /* get 12 clock tick */
    double tick_x = 0.0, tick_y = 0.0, tick_z = 0.0;
    double tick_radius = 0.0;
    double tick_radius_record = 0.0;
    for (i = 0; i < num_points_inlet; i++)
    {
        tick_radius = sqrt(pow((centroids[i][0] - center_x), 2) + pow((centroids[i][1] - center_y), 2) + pow((centroids[i][2] - center_z), 2));
        if (tick_radius > tick_radius_record) {
            tick_x = x[0];
            tick_y = x[1];
            tick_z = x[2];
        }
        tick_radius_record = tick_radius;
    }

    /* start spinning and record maximum radius for every sector */
    int num_sector = 30;
    double angle = 0.0;
    int id_sector = 0;
    double current_radius = 0.0;
    /* tick & tock vector for cross and dot production */
    Vector3D v_tick;
    v_tick.x = tick_x - center_x;
    v_tick.y = tick_y - center_y;
    v_tick.z = tick_z - center_z;
    Vector3D v_tock;
    double radius_list[num_sector];
    for (i = 0; i < num_sector; ++i) {
        radius_list[i] = 0.0;
    }
    /* loop and fill radius_list */
    for (i = 0; i < num_points_inlet; i++)
    {
        v_tock.x = centroids[i][0] - center_x;
        v_tock.y = centroids[i][1] - center_y;
        v_tock.z = centroids[i][2] - center_z;
        angle = getDegAngle(v_tick, v_tock);
        id_sector = floor((angle + PI) / (2 * PI / num_sector));
        if (id_sector < 0) {
            id_sector = 0;
        }
        if (id_sector >= num_sector) {
            id_sector = num_sector - 1;
        }
        current_radius = sqrt(pow(v_tock.x, 2) + pow(v_tock.y, 2) + pow(v_tock.z, 2));
        if (current_radius > radius_list[id_sector]) {
            radius_list[id_sector] = current_radius;
        }
    }

        /* clean radius list and check (sometimes there are no nodes in the sector) */
        for (i = 0; i < num_sector; ++i) {
            if (radius_list[i] == 0.0) {
                radius_list[i] = tick_radius_record;
            }
        }

    /* assign parabolic profile */
    begin_f_loop(f, thread)
    {
        F_CENTROID(x, f, thread);
        /* current_time = CURRENT_TIME */
        int current_time_id = (int)floor(CURRENT_TIME / FIXED_TIME_STEP);
        double current_Q_ratio = Q_array[current_time_id];
        curr_x = x[0]; /* x[0] is x coordinate */
        curr_y = x[1]; /* x[1] is y coordinate */
        curr_z = x[2]; /* x[2] is z coordinate */

        v_tock.x = x[0] - center_x;
        v_tock.y = x[1] - center_y;
        v_tock.z = x[2] - center_z;
        angle = getDegAngle(v_tick, v_tock);
        id_sector = floor((angle + PI) / (2 * PI / num_sector));
        if (id_sector < 0) {
            id_sector = 0;
        }
        if (id_sector >= num_sector) {
            id_sector = num_sector - 1;
        }
        current_radius = radius_list[id_sector];

        /* write inlet velocity profile - need to centre the x,y,z coords and divide by Rref^2 */
        F_PROFILE(f, thread, position) = current_Q_ratio * VINREF * (1 + VIN) * (1 - (pow((curr_x - center_x), 2) + pow((curr_y - center_y), 2) + pow((curr_z - center_z), 2)) / (current_radius * current_radius));
    }
    end_f_loop(f, thread)
}