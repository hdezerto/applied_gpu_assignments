#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

// ------------------------------------------------------------
// GPU implementation of mover_PC
// ------------------------------------------------------------

__device__ __forceinline__ int idx3(const int i, const int j, const int k, const int nyn, const int nzn)
{
    return (i * nyn + j) * nzn + k;
}

__global__ void mover_pc_kernel(
    const int nop,
    FPpart* x, FPpart* y, FPpart* z,
    FPpart* u, FPpart* v, FPpart* w,
    const FPfield* Ex, const FPfield* Ey, const FPfield* Ez,
    const FPfield* Bxn, const FPfield* Byn, const FPfield* Bzn,
    const FPfield* XN, const FPfield* YN, const FPfield* ZN,
    const FPfield invVOL, const FPfield invdx, const FPfield invdy, const FPfield invdz,
    const FPpart qom, const double c, const double dt,
    const int n_sub_cycles, const int NiterMover,
    const double xStart, const double yStart, const double zStart,
    const double Lx, const double Ly, const double Lz,
    const int nxn, const int nyn, const int nzn,
    const int periodicX, const int periodicY, const int periodicZ)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= nop) return;

    FPpart xval = x[pid];
    FPpart yval = y[pid];
    FPpart zval = z[pid];
    FPpart uval = u[pid];
    FPpart vval = v[pid];
    FPpart wval = w[pid];

    const FPpart dt_sub = (FPpart)(dt / (double)n_sub_cycles);
    const FPpart dto2 = (FPpart)(0.5 * dt_sub);
    const FPpart qomdt2 = (FPpart)(qom * dto2 / c);

    for (int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {
        FPpart xptilde = xval;
        FPpart yptilde = yval;
        FPpart zptilde = zval;

        FPpart uptilde = uval;
        FPpart vptilde = vval;
        FPpart wptilde = wval;

        for (int innter = 0; innter < NiterMover; innter++) {
            // interpolation G --> P
            int ix = 2 + int((xval - xStart) * invdx);
            int iy = 2 + int((yval - yStart) * invdy);
            int iz = 2 + int((zval - zStart) * invdz);

            FPfield xi0 = xval - XN[idx3(ix - 1, iy, iz, nyn, nzn)];
            FPfield eta0 = yval - YN[idx3(ix, iy - 1, iz, nyn, nzn)];
            FPfield zeta0 = zval - ZN[idx3(ix, iy, iz - 1, nyn, nzn)];
            FPfield xi1 = XN[idx3(ix, iy, iz, nyn, nzn)] - xval;
            FPfield eta1 = YN[idx3(ix, iy, iz, nyn, nzn)] - yval;
            FPfield zeta1 = ZN[idx3(ix, iy, iz, nyn, nzn)] - zval;

            FPfield weight[2][2][2];
            weight[0][0][0] = xi0 * eta0 * zeta0 * invVOL;
            weight[0][0][1] = xi0 * eta0 * zeta1 * invVOL;
            weight[0][1][0] = xi0 * eta1 * zeta0 * invVOL;
            weight[0][1][1] = xi0 * eta1 * zeta1 * invVOL;
            weight[1][0][0] = xi1 * eta0 * zeta0 * invVOL;
            weight[1][0][1] = xi1 * eta0 * zeta1 * invVOL;
            weight[1][1][0] = xi1 * eta1 * zeta0 * invVOL;
            weight[1][1][1] = xi1 * eta1 * zeta1 * invVOL;

            FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0;
            FPfield Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int nidx = idx3(ix - ii, iy - jj, iz - kk, nyn, nzn);
                        FPfield wght = weight[ii][jj][kk];
                        Exl += wght * Ex[nidx];
                        Eyl += wght * Ey[nidx];
                        Ezl += wght * Ez[nidx];
                        Bxl += wght * Bxn[nidx];
                        Byl += wght * Byn[nidx];
                        Bzl += wght * Bzn[nidx];
                    }

            FPpart omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
            FPpart denom = (FPpart)(1.0 / (1.0 + omdtsq));

            FPpart ut = uval + qomdt2 * Exl;
            FPpart vt = vval + qomdt2 * Eyl;
            FPpart wt = wval + qomdt2 * Ezl;
            FPpart udotb = ut * Bxl + vt * Byl + wt * Bzl;

            uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

            // half-step position update used for next iteration
            xval = xptilde + uptilde * dto2;
            yval = yptilde + vptilde * dto2;
            zval = zptilde + wptilde * dto2;
        }

        // final velocity update
        uval = (FPpart)(2.0 * uptilde - uval);
        vval = (FPpart)(2.0 * vptilde - vval);
        wval = (FPpart)(2.0 * wptilde - wval);

        // full-step position update
        xval = xptilde + uptilde * dt_sub;
        yval = yptilde + vptilde * dt_sub;
        zval = zptilde + wptilde * dt_sub;

        // boundary conditions X
        if (xval > Lx) {
            if (periodicX) {
                xval -= (FPpart)Lx;
            } else {
                uval = -uval;
                xval = (FPpart)(2.0 * Lx - xval);
            }
        }
        if (xval < 0) {
            if (periodicX) {
                xval += (FPpart)Lx;
            } else {
                uval = -uval;
                xval = -xval;
            }
        }

        // boundary conditions Y
        if (yval > Ly) {
            if (periodicY) {
                yval -= (FPpart)Ly;
            } else {
                vval = -vval;
                yval = (FPpart)(2.0 * Ly - yval);
            }
        }
        if (yval < 0) {
            if (periodicY) {
                yval += (FPpart)Ly;
            } else {
                vval = -vval;
                yval = -yval;
            }
        }

        // boundary conditions Z
        if (zval > Lz) {
            if (periodicZ) {
                zval -= (FPpart)Lz;
            } else {
                wval = -wval;
                zval = (FPpart)(2.0 * Lz - zval);
            }
        }
        if (zval < 0) {
            if (periodicZ) {
                zval += (FPpart)Lz;
            } else {
                wval = -wval;
                zval = -zval;
            }
        }
    }

    x[pid] = xval;
    y[pid] = yval;
    z[pid] = zval;
    u[pid] = uval;
    v[pid] = vval;
    w[pid] = wval;
}

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    const int nop = (int)part->nop;
    if (nop == 0) return 0;

    size_t pbytes = (size_t)nop * sizeof(FPpart);
    // device particle arrays
    FPpart *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    FPpart *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;

    cudaMalloc((void**)&d_x, pbytes);
    cudaMalloc((void**)&d_y, pbytes);
    cudaMalloc((void**)&d_z, pbytes);
    cudaMalloc((void**)&d_u, pbytes);
    cudaMalloc((void**)&d_v, pbytes);
    cudaMalloc((void**)&d_w, pbytes);

    cudaMemcpy(d_x, part->x, pbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, part->y, pbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, part->z, pbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, part->u, pbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, part->v, pbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, part->w, pbytes, cudaMemcpyHostToDevice);

    // field and grid arrays (flat)
    const int nxn = grd->nxn;
    const int nyn = grd->nyn;
    const int nzn = grd->nzn;
    size_t nodes_bytes = (size_t)nxn * (size_t)nyn * (size_t)nzn * sizeof(FPfield);

    FPfield *d_Ex = nullptr, *d_Ey = nullptr, *d_Ez = nullptr;
    FPfield *d_Bxn = nullptr, *d_Byn = nullptr, *d_Bzn = nullptr;
    FPfield *d_XN = nullptr, *d_YN = nullptr, *d_ZN = nullptr;

    cudaMalloc((void**)&d_Ex, nodes_bytes);
    cudaMalloc((void**)&d_Ey, nodes_bytes);
    cudaMalloc((void**)&d_Ez, nodes_bytes);
    cudaMalloc((void**)&d_Bxn, nodes_bytes);
    cudaMalloc((void**)&d_Byn, nodes_bytes);
    cudaMalloc((void**)&d_Bzn, nodes_bytes);
    cudaMalloc((void**)&d_XN, nodes_bytes);
    cudaMalloc((void**)&d_YN, nodes_bytes);
    cudaMalloc((void**)&d_ZN, nodes_bytes);

    cudaMemcpy(d_Ex, field->Ex_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ey, field->Ey_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ez, field->Ez_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bxn, field->Bxn_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Byn, field->Byn_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bzn, field->Bzn_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_XN, grd->XN_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_YN, grd->YN_flat, nodes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZN, grd->ZN_flat, nodes_bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 gridDim((nop + block.x - 1) / block.x);

    mover_pc_kernel<<<gridDim, block>>>(
        nop,
        d_x, d_y, d_z, d_u, d_v, d_w,
        d_Ex, d_Ey, d_Ez, d_Bxn, d_Byn, d_Bzn,
        d_XN, d_YN, d_ZN,
        grd->invVOL, grd->invdx, grd->invdy, grd->invdz,
        part->qom, param->c, param->dt,
        part->n_sub_cycles, part->NiterMover,
        grd->xStart, grd->yStart, grd->zStart,
        grd->Lx, grd->Ly, grd->Lz,
        grd->nxn, grd->nyn, grd->nzn,
        param->PERIODICX ? 1 : 0,
        param->PERIODICY ? 1 : 0,
        param->PERIODICZ ? 1 : 0);

    cudaMemcpy(part->x, d_x, pbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, d_y, pbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, d_z, pbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, d_u, pbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, d_v, pbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, d_w, pbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);
    cudaFree(d_Ex); cudaFree(d_Ey); cudaFree(d_Ez);
    cudaFree(d_Bxn); cudaFree(d_Byn); cudaFree(d_Bzn);
    cudaFree(d_XN); cudaFree(d_YN); cudaFree(d_ZN);

    return 0;
}
