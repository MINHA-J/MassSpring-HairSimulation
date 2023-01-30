#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define NUMSTRANDS			10
#define NUMPARTICLES		20 // 이거 수정시 cu도
#define NUMCOMPONENTS		3 // 3D vectors
#define NUMSEGMENTS			(NUMPARTICLES-1)
#define MASS				NUMSTRANDS * 0.02f // ~ 0.000000001f particle mass is 0.01mg, total strand weight is 1mg
#define K_EDGE				6400.0f // ~ (stable value) 10000000.0f
#define K_BEND				160.0f // ~ 8.0f
#define K_TWIST				160.0f // ~ 8.0f
#define K_EXTRA				9.81f // ~ 0.4905f
#define LENGTH				10.0f //0.5f //5 millmetres separation between particles
#define LENGTH_EDGE			LENGTH  // ~ length between edge springs
#define LENGTH_BEND			LENGTH // ~ length between bending springs
#define LENGTH_TWIST		LENGTH  // ~ length between twisting springs
#define LENGTH_EXTRA		LENGTH
#define D_EDGE				64000.f // ~ 3200.0f
#define D_BEND				5000.f // ~ 250.0f
#define D_TWIST				5000.f // ~ 250.0f
#define D_EXTRA				250.f // ~ 12.5f
#define GRAVITY				-9810.f //(stable value) -0.00981f

//Stiction constants
#define K_STIC				0.01f //Stiction spring coefficient
#define D_STIC				0.2f //Stiction damping coefficient
#define	LEN_STIC			LENGTH * 0.7f //Stiction spring rest length (3.5 millimetres)
#define HALF_LEN_STIC		LEN_STIC * 0.5f //Half the sticition spring rest length (for KDOP volume calculation)
#define MAX_LEN_STIC		LENGTH //Maximum length of stiction spring
#define MAX_SQR_STIC		LENGTH*LENGTH //Maximum length of stiction spring squared

//Bounding volume constants
#define KDOP_PLANES			26 //other valid values include 6, 14 & 18.

#define MAX_LENGTH			LENGTH * 1.3f // ~ Maximum length of a spring
#define MAX_LENGTH_SQUARED	MAX_LENGTH*MAX_LENGTH // ~ Maximum length of a spring squared

//Geometry collisions constants
#define DOMAIN_DIM		100
#define DOMAIN_WIDTH	1380.f
#define DOMAIN_HALF		690.f
#define CELL_WIDTH		13.8f
#define CELL_HALF		6.9f

#define POINTS_PER_VERTEX 3
#define TOTAL_FLOATS_IN_TRIANGLE 9
#endif