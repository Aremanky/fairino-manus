#include <stdio.h>
#include <stddef.h>
#include <mujoco/mujoco.h>
#define O(name, expr) printf("\t%s = %zu\n", name, (size_t)(expr))
int main(void){
  printf("// GENERADO por herramientas/offsets.c con las cabeceras de MuJoCo %d (iguales en\n// Windows x64 y Linux x64: solo int, float, double y punteros). No editar a mano.\n\npackage main\n\nconst (\n", mjVERSION_HEADER);
  O("mjVersionCabecera", mjVERSION_HEADER);
  O("tamCamara", sizeof(mjvCamera)); O("tamOpcion", sizeof(mjvOption)); O("tamEscena", sizeof(mjvScene));
  O("tamContexto", sizeof(mjrContext)); O("tamPerturb", sizeof(mjvPerturb)); O("tamGLCamara", sizeof(mjvGLCamera));
  O("mNq", offsetof(mjModel, nq)); O("mNmat", offsetof(mjModel, nmat)); O("mNbody", offsetof(mjModel, nbody));
  O("mNjnt", offsetof(mjModel, njnt)); O("mNmocap", offsetof(mjModel, nmocap));
  O("mOffwidth", offsetof(mjModel, vis.global.offwidth)); O("mOffheight", offsetof(mjModel, vis.global.offheight));
  O("mMatReflectance", offsetof(mjModel, mat_reflectance)); O("mBodyMocapid", offsetof(mjModel, body_mocapid));
  O("mBodyJntnum", offsetof(mjModel, body_jntnum)); O("mBodyJntadr", offsetof(mjModel, body_jntadr));
  O("mJntType", offsetof(mjModel, jnt_type)); O("mJntQposadr", offsetof(mjModel, jnt_qposadr));
  O("cType", offsetof(mjvCamera, type)); O("cLookat", offsetof(mjvCamera, lookat)); O("cDistance", offsetof(mjvCamera, distance));
  O("cAzimuth", offsetof(mjvCamera, azimuth)); O("cElevation", offsetof(mjvCamera, elevation));
  O("sCamera", offsetof(mjvScene, camera));
  O("gPos", offsetof(mjvGLCamera, pos)); O("gForward", offsetof(mjvGLCamera, forward)); O("gUp", offsetof(mjvGLCamera, up));
  O("gFrustumCenter", offsetof(mjvGLCamera, frustum_center)); O("gFrustumBottom", offsetof(mjvGLCamera, frustum_bottom));
  O("gFrustumTop", offsetof(mjvGLCamera, frustum_top)); O("gFrustumNear", offsetof(mjvGLCamera, frustum_near));
  printf(")\n");
  return 0;
}
