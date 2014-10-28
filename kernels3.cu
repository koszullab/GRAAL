#include <curand_kernel.h>

extern "C"
{
    texture<unsigned char, 2> tex;

//    texture<float, 2, cudaReadModeElementType> texData;

    typedef struct frag {
        int* pos;
        int* id_c;
        int* start_bp;
        int* len_bp;
        int* circ;
        int* id;
        int* prev;
        int* next;
        int* l_cont;
        int* l_cont_bp;
        int* ori;
        int* rep;
        int* activ;
        int* id_d;
    } frag;

    typedef struct __attribute__ ((packed)) param_simu {
        float kuhn __attribute__ ((packed));
        float lm __attribute__ ((packed));
        float c1 __attribute__ ((packed));
        float slope __attribute__ ((packed));
        float d __attribute__ ((packed));
        float d_max __attribute__ ((packed));
        float fact __attribute__ ((packed));
        float v_inter __attribute__ ((packed));
    } param_simu;


    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                            __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
    __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
    {
            int id = blockIdx.x*blockDim.x + threadIdx.x;

            if (id >= nthreads)
                    return;
            curand_init(seed, id, offset, &s[id]);
    }

    __global__ void gen_rand_mat(float* initArray, float *randArray, curandState *state, int n_rng, int width,
                                 float fact_sub)
    {
        int r0 = threadIdx.x + blockDim.x * blockIdx.x;
        int r1 = threadIdx.y + blockDim.y * blockIdx.y;
        int coord = r0 * width + r1;
        int coord_y = r1 * width + r0;
        int id_rng = coord % n_rng ;
        float val;
        if ((r0<width) && (r1 < r0)) {
            float mean_exp = initArray[coord] * fact_sub;
            float mean = max(mean_exp, 0.000001);
            val = curand_poisson(&state[id_rng], mean);
            randArray[coord] = val;
            randArray[coord_y] = val;
        }
    }


    __device__ float factorial(float n)
    {

        float result = 1;
        n = floor(n);
        if (n<10){
            for(int c = 1 ; c <= n ; c++ )
                result = result * c;
        }
        else{
            result = powf(n,n) * exp(-n) * sqrtf(2 * M_PI * n);
        }
        return ( result );
    }

     __device__ param_simu modify_param_simu(int id_modifier, param_simu p, float var)
     {
         param_simu out;
         if (id_modifier == 0){
            out.kuhn = p.kuhn;
            out.lm = p.lm;
            out.slope = p.slope;
            out.d = var;
            out.c1 = p.c1;
            out.d_max = p.d_max;
            out.v_inter = p.v_inter;
            out.fact = p.fact;
         }
         else if (id_modifier == 1){
            out.kuhn = p.kuhn;
            out.lm = p.lm;
            out.slope = var;
            out.d = p.d;
            out.c1 = p.c1;
            out.d_max = p.d_max;
            out.v_inter = p.v_inter;
            out.fact = p.fact;
         }
         return (out);
     }
    __device__ float rippe_contacts(float s, param_simu p)
    {
        // s = distance in kb
        // p = model's parameters
        float result = 0.0f;
        if ((s>0.0f) && (s<p.d_max)){
            result = (p.c1 * pow(s, p.slope) * exp((p.d-2)/(pow(s*p.lm/p.kuhn, 2.0f ) + p.d)  )) * p.fact;
        }
        float out = max(result, p.v_inter);
//        else{
//            result = p.v_inter;
//        }
        return ( out );
    }

    __device__ float rippe_contacts_circ(float s, float s_tot, param_simu p)
    {
        // s = distance in kb
        // p = model's parameters
        // s_tot = total length of circular contig
        float result = 0.0f;
        float n_dist = 1.0f;
        float n_tot = 1.0f;
        float n = 1.0f;
        float K = 1.0f;
        float norm_circ, norm_lin, nmax, val;
        if ((s > 0.0f) && (s < p.d_max)){
//        if ((s < s_tot) && (s > 0.0) && (s < p.d_max)){
            K = p.lm / p.kuhn;
            n_dist = s ;
            n_tot = s_tot;
            nmax = K * 1;

            n = K * n_dist *(n_tot - n_dist) / n_tot;

            norm_lin = rippe_contacts(s, p);
            norm_circ = (powf(p.kuhn, -3.0f) * powf(nmax, p.slope) * expf((p.d - 2.0f)/(powf(nmax, 2.0f ) + p.d))) * p.fact;

            val = (powf(p.kuhn, -3.0f) * powf(n, p.slope) * expf((p.d - 2.0f)/(powf(n, 2.0f ) + p.d))) * p.fact;
            result = val * norm_lin / norm_circ;
        }
        float out = max(result, p.v_inter);
//        else{
//            result = p.v_inter;
//        }
        return ( out );
    }


    __device__ float evaluate_likelihood_float(float ex, float ob)
    {
    // ex = expected n contacts
    // ob = observed n contacts
        float res = 0;
        float lim = 15;
        if (ex != 0){
            if (ob >=lim){
               res = ob * logf(ex) - ex - (ob * logf(ob) - ob + logf(sqrtf(ob * 2.0f * M_PI)));
            }
            else if ((ob>0) && (ob<lim)){
                res = ob * logf(ex) - ex - logf(factorial(ob));
             }
            else if (ob==0){
                res = - ex;
            }
        }

        return (res);
    }


    __device__ double evaluate_likelihood_double(double ex, double ob)
    {
    // ex = expected n contacts
    // ob = observed n contacts
        double res = 0;
        double lim = 15;
        if (ex != 0){
            if (ob >=lim){
               res = ob * log(ex) - ex - (ob * log(ob) - ob + log(sqrt(ob * 2.0 * M_PI)));
            }
            else if ((ob>0) && (ob<lim)){
                res = ob * log(ex) - ex - log((double) factorial((float) ob));
             }
            else if (ob==0){
                res = - ex;
            }
        }

        return (res);
    }



    __device__ int2 lin_2_2dpos(int ind)
    {
        int i = ind + 1;
        int x = (-0.5 + 0.5 * sqrt((float) 1 + 8 * (i - 1))) + 2;
        int y =  x * (3 - x) / 2 + i - 1;
        //int2 out = (int2) (x - 1,y - 1);
        int2 out;
        out.x = min(x - 1, y -1);
        out.y = max(x - 1, y - 1);
        return (out);
    }

    __device__ int conv_plan_pos_2_lin(int2 pos)
    {
        int x = pos.x + 1;
        int y = pos.y + 1;
        int i = min(x,y);
        int j = max(x,y);
        int ind = (j * (j - 3)) / 2 + i;
//        int ind = (j * (j - 3) / 2 + i + 1) - 1;
        return ind;
    }



    __global__ void flip_frag(frag* fragArray,frag* o_fragArray, int id_f_flip,
                              int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){

            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];

            fragArray->pos[id_frag] = pos_fi;
            fragArray->id_c[id_frag] = contig_fi;
            fragArray->start_bp[id_frag] = start_bp_fi;
            fragArray->len_bp[id_frag] = len_bp_fi;
            fragArray->circ[id_frag] = circ_fi;
            fragArray->id[id_frag] = id_frag;
            if (id_frag == id_f_flip){
                fragArray->ori[id_frag] = or_fi * -1;
            }
            else{
                fragArray->ori[id_frag] = or_fi;
            }
            fragArray->prev[id_frag] = id_prev_fi;
            fragArray->next[id_frag] = id_next_fi;
            fragArray->l_cont[id_frag] = l_cont_fi;
            fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
            fragArray->rep[id_frag] = rep_fi;
            fragArray->activ[id_frag] = activ_fi;
            fragArray->id_d[id_frag] = id_d_fi;
        }
    }



    __global__ void swap_activity_frag(frag* fragArray,frag* o_fragArray, int id_f_unactiv, int max_id_contig,
                              int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){

            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];

            fragArray->pos[id_frag] = pos_fi;
            fragArray->start_bp[id_frag] = start_bp_fi;
            fragArray->len_bp[id_frag] = len_bp_fi;
            fragArray->circ[id_frag] = circ_fi;
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = or_fi;
//            if ((id_frag == id_f_unactiv) && (id_d_fi != id_frag)){
            if ((id_frag == id_f_unactiv) && (rep_fi == 1)){
                fragArray->activ[id_frag] = 0 * (activ_fi == 1) + 1 * (activ_fi == 0);
                fragArray->id_c[id_frag] = contig_fi * (activ_fi == 1) + (max_id_contig + 1) * (activ_fi == 0);
//                fragArray->id_c[id_frag] = contig_fi * (activ_fi == 1) + (max_id_contig + 1) * (activ_fi == 0);
            }
            else{
                fragArray->activ[id_frag] = activ_fi;
                fragArray->id_c[id_frag] = contig_fi;
            }
            fragArray->prev[id_frag] = id_prev_fi;
            fragArray->next[id_frag] = id_next_fi;
            fragArray->l_cont[id_frag] = l_cont_fi;
            fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
            fragArray->rep[id_frag] = rep_fi;
            fragArray->id_d[id_frag] = id_d_fi;
        }
    }


    __global__ void pop_out_frag(frag* fragArray,frag* o_fragArray, int* pop_id_contigs, int id_f_pop,
                                 int max_id_contig, int n_frags)
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int or_f_pop;
        __shared__ int circ_f_pop;


        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            or_f_pop = o_fragArray->ori[id_f_pop];
            circ_f_pop = o_fragArray->circ[id_f_pop];
        }
        __syncthreads();

        if (id_frag < n_frags){
            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            if (l_cont_f_pop > 2){
                if ( contig_fi == contig_f_pop){
                    if (pos_fi < pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
//                        fragArray->prev[id_frag] = id_prev_fi;
                        if ((id_frag == id_next_f_pop) && (circ_f_pop == 1)){
                            fragArray->prev[id_frag] = id_prev_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        if (pos_fi == (pos_f_pop - 1)){
                            fragArray->next[id_frag] = id_next_f_pop;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
                        fragArray->l_cont[id_frag] = l_cont_fi -1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi == pos_f_pop){
                        fragArray->pos[id_frag] = 0;
                        fragArray->id_c[id_frag] = max_id_contig + 1;
                        pop_id_contigs[id_frag] = max_id_contig + 1;
                        fragArray->start_bp[id_frag] = 0;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = 1;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = 1;
                        fragArray->l_cont_bp[id_frag] = len_bp_fi;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi - 1;
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi - len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == (pos_f_pop + 1)){
                            fragArray->prev[id_frag] = id_prev_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        if ((id_frag == id_prev_f_pop) && (circ_f_pop == 1)){
                            fragArray->next[id_frag] = id_next_f_pop;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
//                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi -1 ;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }

                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    pop_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
            else if (l_cont_f_pop == 2){
                if ( contig_fi == contig_f_pop){
                    if (pos_fi < pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = l_cont_fi -1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi == pos_f_pop){
                        fragArray->pos[id_frag] = 0;
                        fragArray->id_c[id_frag] = max_id_contig + 1;
                        pop_id_contigs[id_frag] = max_id_contig + 1;
                        fragArray->start_bp[id_frag] = 0;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = 1;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = 1;
                        fragArray->l_cont_bp[id_frag] = len_bp_fi;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi - 1;
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi - len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = l_cont_fi -1 ;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }

                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    pop_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->id_c[id_frag] = contig_fi;
                pop_id_contigs[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->rep[id_frag] = rep_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->id_d[id_frag] = id_d_fi;
            }
        }
    }

    __global__ void pop_in_frag_1(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    // split insert @ left
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();


        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                fragArray->pos[id_frag] = 0;
                fragArray->start_bp[id_frag] = 0;
                fragArray->len_bp[id_frag] = len_bp_f_pop;
                fragArray->circ[id_frag] = 0;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = ori_f_pop;
                fragArray->prev[id_frag] = -1;
                fragArray->next[id_frag] = id_f_ins;
                if (circ_f_ins == 0){
                    fragArray->id_c[id_frag] = max_id_contig + 1;
                    fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                }
                else{
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                }
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];

                if (contig_fi == contig_f_ins){
                    if (circ_f_ins == 0){
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == (pos_f_ins -1)){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = pos_f_ins;
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = 1;
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_f_pop;
                            fragArray->next[id_frag] = id_next_f_ins;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins) + 1;
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins) + len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                    }
                    else{ // contig_f_ins is circular
                        if (pos_fi < pos_f_ins){
                                fragArray->pos[id_frag] = l_cont_f_ins - pos_f_ins + pos_fi + 1;
                                fragArray->id_c[id_frag] = contig_f_ins;
                                fragArray->start_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + start_bp_fi + len_bp_f_pop;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_ins - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                                fragArray->rep[id_frag] = rep_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = 1;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_f_ins;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_f_pop;
                            fragArray->next[id_frag] = id_next_f_ins;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - pos_f_ins + 1;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_ins + len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (id_frag == id_prev_f_ins){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
//                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

    __global__ void pop_in_frag_2(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    {
    // split insert @ right
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                if (circ_f_ins == 0){
                    fragArray->pos[id_frag] = pos_f_ins + 1;
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins;
                    fragArray->len_bp[id_frag] = len_bp_f_pop;
                    fragArray->circ[id_frag] = 0;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = ori_f_pop;
                    fragArray->prev[id_frag] = id_f_ins;
                    fragArray->next[id_frag] = -1;
                    fragArray->l_cont[id_frag] = pos_f_ins + 2;
                    fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                    fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                    fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                    fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
                }
                else{
                    fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_f_ins + 1;
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                    + start_bp_f_ins + len_bp_f_ins;
                    fragArray->len_bp[id_frag] = len_bp_f_pop;
                    fragArray->circ[id_frag] = 0;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = ori_f_pop;
                    fragArray->prev[id_frag] = id_f_ins;
                    fragArray->next[id_frag] = -1;
                    fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                    fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                    fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                    fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
                }
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];
                if (contig_fi == contig_f_ins){
                    if(circ_f_ins == 0){
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = pos_f_ins + 2;
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_prev_f_ins;
                            fragArray->next[id_frag] = id_f_pop;
                            fragArray->l_cont[id_frag] = pos_f_ins + 2;
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins + 1);
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == (pos_f_ins + 1)){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - (pos_f_ins + 1);
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                    }
                    else{//circular contig
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_fi;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                            + start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (id_frag == id_next_f_ins){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
//                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_f_ins;
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                            + start_bp_f_ins;
                            fragArray->len_bp[id_frag] = len_bp_f_ins;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_f_ins;
                            fragArray->next[id_frag] = id_f_pop;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins + 1);
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_ins +1){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

    __global__ void pop_in_frag_3(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    // insert frag @ right of id_f_ins
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                fragArray->pos[id_frag] = pos_f_ins + 1;
                fragArray->id_c[id_frag] = contig_f_ins;
                fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins;
                fragArray->len_bp[id_frag] = len_bp_f_pop;
                fragArray->circ[id_frag] = circ_f_ins;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = ori_f_pop;
                fragArray->prev[id_frag] = id_f_ins;
                fragArray->next[id_frag] = id_next_f_ins;
                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];
                if (contig_fi == contig_f_ins){
                    if (pos_fi < pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if ((id_frag == id_next_f_ins) && ( circ_f_ins == 1)){
                            fragArray->prev[id_frag] = id_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi == pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_f_ins;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_f_pop;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi + 1;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi + len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == (pos_f_ins + 1)){
                            fragArray->prev[id_frag] = id_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

    __global__ void pop_in_frag_4(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    // insert frag @ left of id_f_ins
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                fragArray->pos[id_frag] = pos_f_ins ;
                fragArray->id_c[id_frag] = contig_f_ins;
                fragArray->start_bp[id_frag] = start_bp_f_ins;
                fragArray->len_bp[id_frag] = len_bp_f_pop;
                fragArray->circ[id_frag] = circ_f_ins;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = ori_f_pop;
                fragArray->prev[id_frag] = id_prev_f_ins;
                fragArray->next[id_frag] = id_f_ins;
                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];

            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];

                if (contig_fi == contig_f_ins){
                    if (pos_fi < pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        if (pos_fi == pos_f_ins -1){
                            fragArray->next[id_frag] = id_f_pop;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;

                    }
                    else if (pos_fi == pos_f_ins){
                        fragArray->pos[id_frag] = pos_f_ins + 1;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_f_ins;
                        fragArray->prev[id_frag] = id_f_pop;
                        fragArray->next[id_frag] = id_next_f_ins;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;

                    }
                    else if (pos_fi > pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi + 1;
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi + len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;

                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }


    __global__ void split_contig(frag* fragArray,frag* o_fragArray, int* split_id_contigs, int id_f_cut, int upstream,
                                 int max_id_contig, int n_frags)
    {
        __shared__ int contig_f_cut;
        __shared__ int pos_f_cut;
        __shared__ int l_cont_f_cut;
        __shared__ int l_cont_bp_f_cut;
        __shared__ int len_bp_f_cut;
        __shared__ int start_bp_f_cut;
        __shared__ int id_prev_f_cut;
        __shared__ int id_next_f_cut;
        __shared__ int circ_f_cut;
        __shared__ int or_f_cut;
        __shared__ int activ_f_cut;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_cut = o_fragArray->id_c[id_f_cut];
            pos_f_cut = o_fragArray->pos[id_f_cut];
            l_cont_f_cut = o_fragArray->l_cont[id_f_cut];
            l_cont_bp_f_cut = o_fragArray->l_cont_bp[id_f_cut];
            len_bp_f_cut = o_fragArray->len_bp[id_f_cut];
            start_bp_f_cut = o_fragArray->start_bp[id_f_cut];
            id_prev_f_cut = o_fragArray->prev[id_f_cut];
            id_next_f_cut = o_fragArray->next[id_f_cut];
            circ_f_cut = o_fragArray->circ[id_f_cut];
            or_f_cut = o_fragArray->ori[id_f_cut];
            activ_f_cut = o_fragArray->activ[id_f_cut];
        }
        __syncthreads();
        if (id_frag < n_frags){
            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];
            if ((activ_f_cut == 1) && (l_cont_f_cut > 1)){
                if (contig_fi == contig_f_cut){
                    if (circ_f_cut == 0){ // linear contig
                        if (upstream == 1){
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_cut - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if(pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = 0;
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig + 1;
                                fragArray->start_bp[id_frag] = 0;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = -1;
                                fragArray->next[id_frag] = id_next_f_cut;
                                fragArray->l_cont[id_frag] = l_cont_f_cut  - pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - pos_f_cut;
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig + 1;
                                fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut  - pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                        else{
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = pos_f_cut + 1;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut + len_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if(pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = pos_f_cut;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_f_cut;
                                fragArray->next[id_frag] = -1;
                                fragArray->l_cont[id_frag] = pos_f_cut + 1;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut + len_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - (pos_f_cut + 1);
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig +1;
                                fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                if (pos_fi == pos_f_cut + 1){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut - (pos_f_cut + 1);
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                    }
                    else{ // circular contig !!! problem to correct !!!
                        if (upstream ==1){
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = l_cont_f_cut - pos_f_cut + pos_fi;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut + start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_cut - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = 0;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = 0;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = -1;
                                fragArray->next[id_frag] = id_next_f_cut;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - pos_f_cut;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (id_frag == id_prev_f_cut){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
//                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                        else{
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = (l_cont_f_cut - (pos_f_cut  + 1)) + pos_fi;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = (l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut))
                                                                + start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
//                                fragArray->prev[id_frag] = id_prev_fi;
                                if (id_frag == id_next_f_cut){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = (l_cont_f_cut - (pos_f_cut  + 1)) + pos_fi;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = (l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut))
                                                                + start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_f_cut;
                                fragArray->next[id_frag] = -1;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - (pos_f_cut + 1);
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                if (pos_fi == pos_f_cut +1){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->id_c[id_frag] = contig_fi;
                    split_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;
                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->id_c[id_frag] = contig_fi;
                split_id_contigs[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->id_d[id_frag] = id_d_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->rep[id_frag] = rep_fi;
            }
        }
    }

    __global__ void paste_contigs(frag* fragArray,frag* o_fragArray, int id_fA, int id_fB, int max_id_contig,
                                  int n_frags)
    {
        __shared__ int contig_fA;
        __shared__ int pos_fA;
        __shared__ int l_cont_fA;
        __shared__ int l_cont_bp_fA;
        __shared__ int len_bp_fA;
        __shared__ int start_bp_fA;
        __shared__ int id_prev_fA;
        __shared__ int id_next_fA;
        __shared__ int circ_fA;
        __shared__ int or_fA;
        __shared__ int activ_fA;

        __shared__ int contig_fB;
        __shared__ int pos_fB;
        __shared__ int l_cont_fB;
        __shared__ int l_cont_bp_fB;
        __shared__ int len_bp_fB;
        __shared__ int start_bp_fB;
        __shared__ int id_prev_fB;
        __shared__ int id_next_fB;
        __shared__ int circ_fB;
        __shared__ int or_fB;
        __shared__ int activ_fB;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_fA = o_fragArray->id_c[id_fA];
            pos_fA = o_fragArray->pos[id_fA];
            l_cont_fA = o_fragArray->l_cont[id_fA];
            l_cont_bp_fA = o_fragArray->l_cont_bp[id_fA];
            len_bp_fA = o_fragArray->len_bp[id_fA];
            start_bp_fA = o_fragArray->start_bp[id_fA];
            id_prev_fA = o_fragArray->prev[id_fA];
            id_next_fA = o_fragArray->next[id_fA];
            circ_fA = o_fragArray->circ[id_fA];
            or_fA = o_fragArray->ori[id_fA];
            activ_fA = o_fragArray->activ[id_fA];

            contig_fB = o_fragArray->id_c[id_fB];
            pos_fB = o_fragArray->pos[id_fB];
            l_cont_fB = o_fragArray->l_cont[id_fB];
            l_cont_bp_fB = o_fragArray->l_cont_bp[id_fB];
            len_bp_fB = o_fragArray->len_bp[id_fB];
            start_bp_fB = o_fragArray->start_bp[id_fB];
            id_prev_fB = o_fragArray->prev[id_fB];
            id_next_fB = o_fragArray->next[id_fB];
            circ_fB = o_fragArray->circ[id_fB];
            or_fB = o_fragArray->ori[id_fB];
            activ_fB = o_fragArray->activ[id_fB];
        }
        __syncthreads();

        int contig_fi = o_fragArray->id_c[id_frag];
        int pos_fi = o_fragArray->pos[id_frag];
        int l_cont_fi = o_fragArray->l_cont[id_frag];
        int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
        int len_bp_fi = o_fragArray->len_bp[id_frag];
        int circ_fi = o_fragArray->circ[id_frag];
        int id_prev_fi = o_fragArray->prev[id_frag];
        int id_next_fi = o_fragArray->next[id_frag];
        int start_bp_fi = o_fragArray->start_bp[id_frag];
        int or_fi = o_fragArray->ori[id_frag];
        int rep_fi = o_fragArray->rep[id_frag];
        int activ_fi = o_fragArray->activ[id_frag];
        int id_d_fi = o_fragArray->id_d[id_frag];

        if (id_frag < n_frags){
            if ( (activ_fA == 1) && ( activ_fB == 1) ){
                if (contig_fA != contig_fB){
                    if (contig_fi == contig_fA){
                        if (pos_fA == 0){
                            fragArray->pos[id_frag] = l_cont_fA - (pos_fi + 1);
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA - (start_bp_fi + len_bp_fi);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi * -1;
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_next_fi;
                            }
                            if (pos_fi == pos_fA){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_prev_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;


                        }
                        else{
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == pos_fA){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                    }
                    else if (contig_fi == contig_fB){
                        if (pos_fB == 0){
                            fragArray->pos[id_frag] = l_cont_fA + pos_fi;
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA + start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else{
                            fragArray->pos[id_frag] = l_cont_fA + (l_cont_fB - (pos_fi + 1));
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA + (l_cont_bp_fB - (start_bp_fi + len_bp_fi));
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi * -1;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_next_fi;
                            }
                            if (pos_fi == 0){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_prev_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                    }
                    else{
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;
                    }

                }
                else if (contig_fA == contig_fB){ // circular contig
                    if (contig_fi == contig_fA){
                        if ((pos_fA == 0) && (pos_fB == l_cont_fA - 1)){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_fi;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 1;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->id[id_frag] = id_frag;
                            if (pos_fi == pos_fA){
                                fragArray->prev[id_frag] = id_fB;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->next[id_frag] = id_fA;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                        else if((pos_fA == l_cont_fA - 1) && (pos_fB == 0)){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->id_c[id_frag] = contig_fi;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->circ[id_frag] = 1;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                    }
                    else{
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->id_c[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;
                    }

                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->id_c[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->id_d[id_frag] = id_d_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->rep[id_frag] = rep_fi;
            }
        }
    }

    __global__ void fill_2d_contacts(float*  simuData, float* simu2dData, int max_id, int width_mat)
    {

        int id_pix_out = blockIdx.x * blockDim.x + threadIdx.x;
        for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
            int2 pos_frag = lin_2_2dpos(id_pix);
            int x = min(pos_frag.x,pos_frag.y);
            int y = max(pos_frag.x,pos_frag.y);
            simu2dData[x * width_mat + y] = simuData[id_pix];
        }
    }




//     __global__ void simulate_data_2d(float* obsData2D, float* sub_obsData2D, frag* fragArray,
//                                      int4* id_sub_frags, float3* len_bp_sub_frags,
//                                      param_simu* P,
//                                      int max_id,
//                                      int width_matrix, int width_sub_matrix,
//                                      curandState* state, int n_rng)
//     {
//        param_simu p = P[0];
//        float val_out = 0;
//        float obsD;
//        float s_tot, s, is, is0, is1, is2 = 0;
//        float val_trans, val_cis = 0;
//        float expected_cis  = 0;
//        int contig_i, contig_j;
//        int pos_i, pos_j, or_fi, or_fj, fi, fj, start_idx_i, start_idx_j, end_idx_i, end_idx_j;
//        int2 pos_frag;
//        float fi_start_bp, fj_start_bp;
//        int4 id_sub_frags_fi, id_sub_frags_fj;
//        float3 len_bp_sub_frags_fi, len_bp_sub_frags_fj;
//        float tmp_start_bp_fi, tmp_start_bp_fj;
//        float len_sub_fi[3] = {0.0f,0.0f,0.0f};
//        float len_sub_fj[3] = {0.0f,0.0f,0.0f};
//        int idx_sub_fi[3] = {0,0,0};
//        int idx_sub_fj[3] = {0,0,0};
//        float list_s_fi[3] = {0.0f,0.0f,0.0f};
//        float list_s_fj[3] = {0.0f,0.0f,0.0f};
//        int list_id_data_i[3] = {0,0,0};
//        int list_id_data_j[3] = {0,0,0};
//        int limit_fi = 0;
//        int limit_fj = 0;
//        float accu_i = 0.0f;
//        float accu_j = 0.0f;
//
//        int id_pix_out = blockIdx.x * blockDim.x + threadIdx.x;
//        int i,j;
//        float tmp_val = 0;
//        int id_rng;
//        for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
//             id_rng = id_pix % n_rng;
//             val_out = 0.0;
//             tmp_val = 0.0;
//             pos_frag = lin_2_2dpos(id_pix);
//             fi = pos_frag.x; // frag_i absolute id
//             fj = pos_frag.y; // frag_j absolute id
//             expected_cis  = 0.0f;
//             val_trans, val_cis = 0.0f;
//             contig_i = fragArray->id_c[fi];
//             contig_j = fragArray->id_c[fj];
//             if (contig_i == contig_j){
//                 pos_i = fragArray->pos[fi];
//                 pos_j = fragArray->pos[fj];
//                 if (pos_i > pos_j){ // fi is always the closest frag to the origin
//                    fi = pos_frag.y;
//                    fj = pos_frag.x;
//                 }
//                 or_fi = fragArray->ori[fi];
//                 or_fj = fragArray->ori[fj];
//                 s_tot = int2float(fragArray->l_cont_bp[fi]) / 1000.0f;
//
//                 fi_start_bp =  int2float(fragArray->start_bp[fi]);
//                 fj_start_bp =  int2float(fragArray->start_bp[fj]);
//
//                 id_sub_frags_fi = id_sub_frags[fi];
//                 id_sub_frags_fj = id_sub_frags[fj];
//                 len_bp_sub_frags_fi = len_bp_sub_frags[fi];
//                 len_bp_sub_frags_fj = len_bp_sub_frags[fj];
//
//                 len_sub_fi[0] = len_bp_sub_frags_fi.x;
//                 len_sub_fi[1] = len_bp_sub_frags_fi.y;
//                 len_sub_fi[2] = len_bp_sub_frags_fi.z;
//
//                 len_sub_fj[0] = len_bp_sub_frags_fj.x;
//                 len_sub_fj[1] = len_bp_sub_frags_fj.y;
//                 len_sub_fj[2] = len_bp_sub_frags_fj.z;
//
//                 idx_sub_fi[0] = id_sub_frags_fi.x;
//                 idx_sub_fi[1] = id_sub_frags_fi.y;
//                 idx_sub_fi[2] = id_sub_frags_fi.z;
//
//                 idx_sub_fj[0] = id_sub_frags_fj.x;
//                 idx_sub_fj[1] = id_sub_frags_fj.y;
//                 idx_sub_fj[2] = id_sub_frags_fj.z;
//
//
//                ////////////// under construction ///////////////////
//                 limit_fi = id_sub_frags_fi.w - 1;
//                 limit_fj = id_sub_frags_fj.w - 1;
//
//                 if (or_fi == 1){
//                    accu_i = fi_start_bp/1000.0f + len_sub_fi[0];
//                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[0] / 2.0f;
//                    list_id_data_i[0] = idx_sub_fi[0];
//                    for (int i = 1; i <= limit_fi; i++){
//                        list_s_fi[i] = accu_i + len_sub_fi[i] / 2.0f;
//                        accu_i = accu_i + len_sub_fi[i];
//                        list_id_data_i[i] = idx_sub_fi[i];
//                    }
//                 }
//                 else{
//                    accu_i = fi_start_bp/1000.0f + len_sub_fi[limit_fi];
//                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[limit_fi] / 2.0f;
//                    list_id_data_i[0] = idx_sub_fi[limit_fi];
//                    for (i = 1; i <= limit_fi; i++){
//                        list_s_fi[i] = accu_i + len_sub_fi[limit_fi - i] / 2.0f;
//                        accu_i = accu_i + len_sub_fi[limit_fi -i];
//                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
//
//                    }
//                 }
//                 if (or_fj == 1){
//                    accu_j = fj_start_bp/1000.0f + len_sub_fj[0];
//                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[0] / 2.0f;
//                    list_id_data_j[0] = idx_sub_fj[0];
//                    for (int j = 1; j <= limit_fj; j++){
//                        list_s_fj[j] = accu_j + len_sub_fj[j] / 2.0f;
//                        accu_j = accu_j + len_sub_fj[j];
//                        list_id_data_j[j] = idx_sub_fj[j];
//                    }
//                 }
//                 else{
//                    accu_j = fj_start_bp/1000.0f + len_sub_fj[limit_fj];
//                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[limit_fj] / 2.0f;
//                    list_id_data_j[0] = idx_sub_fj[limit_fj];
//                    for (j = 1; j <= limit_fj; j++){
//                        list_s_fj[j] = accu_j + len_sub_fj[limit_fj - j] / 2.0f;
//                        accu_j = accu_j + len_sub_fj[limit_fj -j];
//                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
//                    }
//                 }
//
//                 for (i = 0; i <= limit_fi; i ++){
//                    for (j = 0; j <=  limit_fj; j ++){
//                         s = fabsf(list_s_fj[j] - list_s_fi[i]);
//                         if (fragArray->circ[fi] == 1){
//                             expected_cis =  rippe_contacts_circ(s, s_tot, p);
//                         }
//                         else{
//                             expected_cis =  rippe_contacts(s, p);
//                         }
//                         tmp_val = tmp_val + expected_cis;
//                         obsData2D[list_id_data_i[i] * width_matrix + list_id_data_j[j]] = expected_cis;
//                    }
//                 }
//                ////////////// end construction ///////////////////
//                 val_out = (float) curand_poisson(&state[id_rng], tmp_val);
//             }
//             else{
//                 or_fi = fragArray->ori[fi];
//                 or_fj = fragArray->ori[fj];
//                 id_sub_frags_fi = id_sub_frags[fi];
//                 id_sub_frags_fj = id_sub_frags[fj];
//
//                 idx_sub_fi[0] = id_sub_frags_fi.x;
//                 idx_sub_fi[1] = id_sub_frags_fi.y;
//                 idx_sub_fi[2] = id_sub_frags_fi.z;
//
//                 idx_sub_fj[0] = id_sub_frags_fj.x;
//                 idx_sub_fj[1] = id_sub_frags_fj.y;
//                 idx_sub_fj[2] = id_sub_frags_fj.z;
//
//                ////////////// under construction ///////////////////
//                 limit_fi = id_sub_frags_fi.w - 1;
//                 limit_fj = id_sub_frags_fj.w - 1;
//
//                 if (or_fi == 1){
//                    list_id_data_i[0] = idx_sub_fi[0];
//                    for (i = 1; i <= limit_fi; i++){
//                        list_id_data_i[i] = idx_sub_fi[i];
//                    }
//                 }
//                 else{
//                    list_id_data_i[0] = idx_sub_fi[limit_fi];
//                    for (i = 1; i <= limit_fi; i++){
//                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
//                    }
//                 }
//                 if (or_fj == 1){
//                    list_id_data_j[0] = idx_sub_fj[0];
//                    for (j = 1; j <= limit_fj; j++){
//                        list_id_data_j[j] = idx_sub_fj[j];
//                    }
//                 }
//                 else{
//                    list_id_data_j[0] = idx_sub_fj[limit_fj];
//                    for (j = 1; j <= limit_fj; j++){
//                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
//                    }
//                 }
//
//                 for (i = 0; i <= limit_fi; i ++){
//                    for (j = 0; j <=  limit_fj; j ++){
//                         tmp_val = tmp_val + p.v_inter;
//                         obsData2D[list_id_data_i[i] * width_matrix + list_id_data_j[j]] = p.v_inter;
//                    }
//                 }
//
//                ////////////// end construction ///////////////////
//                 val_out = (float) curand_poisson(&state[id_rng], tmp_val);
//             }
//             sub_obsData2D[fi * width_sub_matrix + fj] = val_out;
//             sub_obsData2D[fj * width_sub_matrix + fi] = val_out;
//        }
//    }

//    __global__ void simulate_data(float* simuData, frag* fragArray, int max_id, param_simu* P,
//                                  curandState* state, int n_rng)
//    {
//        int id_pix = threadIdx.x + blockDim.x * blockIdx.x;
//        if (id_pix< max_id){
//            param_simu p = P[0];
//            int2 pos_frag = lin_2_2dpos(id_pix);
//            float fi_start_bp =  int2float(fragArray->start_bp[pos_frag.x]);
//            float fj_start_bp =  int2float(fragArray->start_bp[pos_frag.y]);
//            float fi_len_bp = int2float(fragArray->len_bp[pos_frag.x]);
//            float fj_len_bp = int2float(fragArray->len_bp[pos_frag.y]);
//            int fi_pos = fragArray->pos[pos_frag.x];
//            int fj_pos = fragArray->pos[pos_frag.y];
//            int id_cont_i = fragArray->id_c[pos_frag.x];
//            int id_cont_j = fragArray->id_c[pos_frag.y];
//            int id_rng = id_pix % n_rng ;
//            float mean = 0;
//            float s0, s1, s2, s = 0;
//
//            if ((id_cont_i == id_cont_j)){
//                s0 = (fi_len_bp + fj_len_bp) / 2000.;
//                s1 = 0;
//                if ( fi_pos<fj_pos){
//                    s1 = (fi_len_bp / 1000.);
//                }
//                else{
//                    s1 = (fj_len_bp / 1000.);
//                }
//                s2 = fabsf( fi_start_bp - fj_start_bp) / 1000.;
//                s = s0 - s1 + s2;
//                mean = rippe_contacts(s, p);
//            }
//            else{
//                s = -1;
//                mean = p.v_inter;
//            }
//            simuData[id_pix] = (float) curand_poisson(&state[id_rng], mean);
//        }
//    }

     __global__ void simulate_data_2d(frag* fragArray,
                                      int* collector_id,
                                      int2* dispatcher,
                                      int4* id_sub_frags,
                                      int4* rep_id_sub_frags,
                                      float3* len_bp_sub_frags,
                                      int3* accu_sub_frags,
                                      float * exp_mat,
                                      float* full_exp_mat,
                                      param_simu* P,
                                      int max_id_up_diag,
                                      int max_id,
                                      int n_bins,
                                      int width_matrix,
                                      float n_frags_per_bins, curandState* state, int n_rng)
     {
        param_simu p = P[0];
        float val_observed = 0.0;
        float val_expected = 0.0;
        float s_tot, s, is, is0, is1, is2 = 0;
        float val_trans, val_cis = 0;
        float expected_cis  = 0.0f;
        float expected_trans  = 0.0f;
        int contig_i, contig_j;
        int pos_i, pos_j, or_fi, or_fj, fi, fj, start_idx_i, start_idx_j, end_idx_i, end_idx_j;
        int2 pos_frag;
        float fi_start_bp, fj_start_bp;
        int3 accu_sub_frags_fi, accu_sub_frags_fj;
        int4 id_sub_frags_fi, id_sub_frags_fj;
        int4 rep_id_sub_frags_fi, rep_id_sub_frags_fj;
        float3 len_bp_sub_frags_fi, len_bp_sub_frags_fj;
        float tmp_start_bp_fi, tmp_start_bp_fj;
        float len_sub_fi[3] = {0.0f,0.0f,0.0f};
        float len_sub_fj[3] = {0.0f,0.0f,0.0f};
        int idx_sub_fi[3] = {0,0,0};
        int idx_sub_fj[3] = {0,0,0};
        int rep_idx_sub_fi[3] = {0,0,0};
        int rep_idx_sub_fj[3] = {0,0,0};
        int accu_sub_fi[3] = {0,0,0};
        int accu_sub_fj[3] = {0,0,0};
        float list_s_fi[3] = {0.0f,0.0f,0.0f};
        float list_s_fj[3] = {0.0f,0.0f,0.0f};
        int list_id_data_i[3] = {0,0,0};
        int list_id_data_j[3] = {0,0,0};
        int rep_list_id_data_i[3] = {0,0,0};
        int rep_list_id_data_j[3] = {0,0,0};
        int list_accu_data_i[3] = {0,0,0};
        int list_accu_data_j[3] = {0,0,0};

        float local_storage_exp[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        float local_storage_obs[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        int local_storage_id[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        int local_sub_pos_fi[3] = {0,1,2};
        int local_sub_pos_fj[3] = {0,1,2};
        int id_data_fi = 0;
        int id_data_fj = 0;
        int o_id_data_fi = 0;
        int o_id_data_fj = 0;
        int tmp_id_data = 0;
        int init_limit_fi, init_limit_fj;
        int id_rep_fi, id_rep_fj;
        int2 data_pos, dispatch_fi, dispatch_fj, full_id_2d, pos_2d_sub_pxl , rep_pos_2d_sub_pxl;
        int sub_pixel_coord = 0;
        int rep_sub_pixel_coord = 0;
        int swap = 0;
        int limit_fi = 0;
        int limit_fj = 0;
        float accu_i = 0.0f;
        float accu_j = 0.0f;
        int i,j;
        float norm_accu = 1.0;
        float val_pix = 0.0f;
        int id_pix_2_write, tmp_id_fi_fj, tmp_id_data_fi_fj;
        int id_pix_out = blockIdx.x * blockDim.x + threadIdx.x;
//        int full_id_pix_2_write;
        float cumul_exp = 0.0f;
        int is_activ_fi, is_activ_fj;
        int is_rep_fi, is_rep_fj;
        int are_the_same = 0;
        int on_diag = 0;
        int loop_id_i = 0;
        int loop_id_j = 0;
        int id_rng;
        for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
            id_rng = id_pix % n_rng;
            if (id_pix < max_id_up_diag){
                pos_frag = lin_2_2dpos(id_pix);
                o_id_data_fi = pos_frag.x; // frag_i absolute id
                o_id_data_fj = pos_frag.y; // frag_j absolute id
                on_diag = 0;
            }
            else{
                o_id_data_fi = (id_pix - max_id_up_diag) % n_bins;
                o_id_data_fj = o_id_data_fi;
                on_diag = 1;
            }
            id_pix_2_write = o_id_data_fi * n_bins + o_id_data_fj;
            val_pix = 0.0f;
            dispatch_fi = dispatcher[o_id_data_fi];
            dispatch_fj = dispatcher[o_id_data_fj];

            for (i=0;i<3;i++){
                for(j=0;j<3;j++){
                local_storage_exp[i][j] = 0.0f;
                }
            }
            init_limit_fi = id_sub_frags[fragArray->id_d[collector_id[dispatch_fi.x]]].w - 1;
            init_limit_fj = id_sub_frags[fragArray->id_d[collector_id[dispatch_fj.x]]].w - 1;
            ///// collect observed data ////
            loop_id_i = 0;
            loop_id_j = 0;
            for(id_rep_fi = dispatch_fi.x; id_rep_fi < dispatch_fi.y; id_rep_fi ++){
                fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                is_activ_fi = fragArray->activ[fi];
                is_rep_fi = fragArray->rep[fi];
                if (is_activ_fi == 1){
                    for(id_rep_fj = dispatch_fj.x; id_rep_fj< dispatch_fj.y; id_rep_fj ++){
                         fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                         fj = collector_id[id_rep_fj]; // index (with rep) frag curr_level
                         is_activ_fj = fragArray->activ[fj];
                         is_rep_fj = fragArray->rep[fj];
                         swap = 0;
                         are_the_same = (fi == fj);
                         if ((is_activ_fj == 1)){
//                         if ((is_activ_fj == 1) && ( (are_the_same == 0) || ((is_rep_fi == 0) && (is_rep_fj == 0)))){
//                         if ((is_activ_fj == 1) && ( are_the_same == 0)){
                             norm_accu = 1.0f;

                             id_data_fi = fragArray->id_d[fi];
                             id_data_fj = fragArray->id_d[fj];
                             data_pos.x = id_data_fi;
                             data_pos.y = id_data_fj;
                             expected_cis  = 0.0f;
                             val_trans, val_cis = 0.0f;
                             contig_i = fragArray->id_c[fi];
                             contig_j = fragArray->id_c[fj];
                             if (contig_i == contig_j){
                                 pos_i = fragArray->pos[fi];
                                 pos_j = fragArray->pos[fj];
                                 if (pos_i > pos_j){ // fi is always the closest frag to the origin
                                    swap = 1;
                                    tmp_id_fi_fj = fi;
                                    fi = fj;
                                    fj = tmp_id_fi_fj;

                                    tmp_id_data_fi_fj = id_data_fi;
                                    id_data_fi = id_data_fj;
                                    id_data_fj = tmp_id_data_fi_fj;
                                 }
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];
                                 s_tot = int2float(fragArray->l_cont_bp[fi]) / 1000.0f;

                                 fi_start_bp =  int2float(fragArray->start_bp[fi]);
                                 fj_start_bp =  int2float(fragArray->start_bp[fj]);

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];
                                 //////// rep id sub frags ///////
//                                 rep_id_sub_frags_fi = rep_id_sub_frags[fi];
//                                 rep_id_sub_frags_fj = rep_id_sub_frags[fj];
                                 /////////////////////////////////
                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];

                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 len_sub_fi[0] = len_bp_sub_frags_fi.x;
                                 len_sub_fi[1] = len_bp_sub_frags_fi.y;
                                 len_sub_fi[2] = len_bp_sub_frags_fi.z;

                                 len_sub_fj[0] = len_bp_sub_frags_fj.x;
                                 len_sub_fj[1] = len_bp_sub_frags_fj.y;
                                 len_sub_fj[2] = len_bp_sub_frags_fj.z;

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;
                                 //////// rep id sub frags ///////
//                                 rep_idx_sub_fi[0] = rep_id_sub_frags_fi.x;
//                                 rep_idx_sub_fi[1] = rep_id_sub_frags_fi.y;
//                                 rep_idx_sub_fi[2] = rep_id_sub_frags_fi.z;
//
//                                 rep_idx_sub_fj[0] = rep_id_sub_frags_fj.x;
//                                 rep_idx_sub_fj[1] = rep_id_sub_frags_fj.y;
//                                 rep_idx_sub_fj[2] = rep_id_sub_frags_fj.z;
                                 //////////////////////////////////


                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[0];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[0] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[0];
//                                    rep_list_id_data_i[0] = rep_idx_sub_fi[0];
                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[i];
                                        list_id_data_i[i] = idx_sub_fi[i];
//                                        rep_list_id_data_i[i] = rep_idx_sub_fi[i];
                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[limit_fi];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[limit_fi] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];
//                                    rep_list_id_data_i[0] = rep_idx_sub_fi[limit_fi];
                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[limit_fi - i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[limit_fi -i];
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
//                                        rep_list_id_data_i[i] = rep_idx_sub_fi[limit_fi - i];
                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi - i];
                                    }
                                 }
                                 if (or_fj == 1){
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[0];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[0] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[0];
//                                    rep_list_id_data_j[0] = rep_idx_sub_fj[0];
                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[j];
                                        list_id_data_j[j] = idx_sub_fj[j];
//                                        rep_list_id_data_j[j] = rep_idx_sub_fj[j];
                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[limit_fj];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[limit_fj] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];
//                                    rep_list_id_data_j[0] = rep_idx_sub_fj[limit_fj];
                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[limit_fj - j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[limit_fj -j];
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
//                                        rep_list_id_data_j[j] = rep_idx_sub_fj[limit_fj - j];
                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i = 0; i <= limit_fi; i ++){
                                    for (j = 0 ; j <=  limit_fj; j ++){
                                         s = fabsf(list_s_fj[j] - list_s_fi[i]);
                                         ////////////////////////////////////////////
//                                         pos_2d_sub_pxl.x = list_id_data_i[i];
//                                         pos_2d_sub_pxl.y = list_id_data_j[j];
//                                         sub_pixel_coord = conv_plan_pos_2_lin(pos_2d_sub_pxl); // coord data && uniq frag
                                         sub_pixel_coord = min(list_id_data_i[i], list_id_data_j[j]) * width_matrix +
                                                                 max(list_id_data_i[i], list_id_data_j[j]); // coord data && uniq frag

                                         ////////////////////////////////////////////
//                                         rep_pos_2d_sub_pxl.x = rep_list_id_data_i[i];
//                                         rep_pos_2d_sub_pxl.y = rep_list_id_data_j[j];
//                                         rep_sub_pixel_coord = conv_plan_pos_2_lin(rep_pos_2d_sub_pxl); // coord repeated frag
                                         ////////////////////////////////////////////
                                         norm_accu = int2float(list_accu_data_i[i] * list_accu_data_j[j]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         if (fragArray->circ[fi] == 1){
                                             expected_cis =  rippe_contacts_circ(s, s_tot, p) * norm_accu;
                                         }
                                         else{
                                             expected_cis =  rippe_contacts(s, p) * norm_accu;
                                         }
                                         // local storage of inter sub frags contacts
                                         if (swap == 0){
                                            val_cis = local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]];
                                            local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = val_cis +expected_cis;

                                            if ((loop_id_i ==0) && (loop_id_j ==0)){
                                                local_storage_id[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = sub_pixel_coord;
                                            }
                                         }
                                         else{
                                            val_cis = local_storage_exp[local_sub_pos_fj[j]][local_sub_pos_fi[i]];
                                            local_storage_exp[local_sub_pos_fj[j]][local_sub_pos_fi[i]] = val_cis + expected_cis;

                                            if ((loop_id_i ==0) && (loop_id_j ==0)){
                                                local_storage_id[local_sub_pos_fj[j]][local_sub_pos_fi[i]] = sub_pixel_coord;
                                            }
                                         }
                                         // storage of inter sub frags contacts
//                                         full_exp_mat_rep[rep_sub_pixel_coord] = expected_cis;
//                                         atomicAdd(&full_exp_mat[min(list_id_data_i[i], list_id_data_j[j]) * width_matrix +
//                                                                 max(list_id_data_i[i], list_id_data_j[j])],
//                                                                 (float) curand_poisson(&state[id_rng],expected_cis)); // accumulation of sub frags
                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             else{
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];
                                 //////// rep id sub frags ///////
//                                 rep_id_sub_frags_fi = rep_id_sub_frags[fi];
//                                 rep_id_sub_frags_fj = rep_id_sub_frags[fj];
                                 /////////////////////////////////
                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];
                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;
                                 //////// rep id sub frags ///////
//                                 rep_idx_sub_fi[0] = rep_id_sub_frags_fi.x;
//                                 rep_idx_sub_fi[1] = rep_id_sub_frags_fi.y;
//                                 rep_idx_sub_fi[2] = rep_id_sub_frags_fi.z;
//
//                                 rep_idx_sub_fj[0] = rep_id_sub_frags_fj.x;
//                                 rep_idx_sub_fj[1] = rep_id_sub_frags_fj.y;
//                                 rep_idx_sub_fj[2] = rep_id_sub_frags_fj.z;
                                 //////////////////////////////////
                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    list_id_data_i[0] = idx_sub_fi[0];
//                                    rep_list_id_data_i[0] = rep_idx_sub_fi[0];
                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[i];
//                                        rep_list_id_data_i[i] = rep_idx_sub_fi[i];
                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];
//                                    rep_list_id_data_i[0] = rep_idx_sub_fi[limit_fi];
                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];

                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
//                                        rep_list_id_data_i[i] = rep_idx_sub_fi[limit_fi - i];
                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi];

                                    }
                                 }
                                 if (or_fj == 1){
                                    list_id_data_j[0] = idx_sub_fj[0];
//                                    rep_list_id_data_j[0] = rep_idx_sub_fj[0];
                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[j];
//                                        rep_list_id_data_j[j] = rep_idx_sub_fj[j];
                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];
//                                    rep_list_id_data_j[0] = rep_idx_sub_fj[limit_fj];
                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
//                                        rep_list_id_data_j[j] = rep_idx_sub_fj[limit_fj - j];
                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i = 0; i <= limit_fi; i ++){
                                    for (j = 0 ; j <=  limit_fj; j ++){
                                         ////////////////////////////////////////////
//                                         pos_2d_sub_pxl.x = list_id_data_i[i];
//                                         pos_2d_sub_pxl.y = list_id_data_j[j];
//                                         sub_pixel_coord = conv_plan_pos_2_lin(pos_2d_sub_pxl); // coord data && uniq frag
                                         sub_pixel_coord = min(list_id_data_i[i], list_id_data_j[j]) * width_matrix +
                                                                 max(list_id_data_i[i], list_id_data_j[j]); // coord data && uniq frag
                                         ////////////////////////////////////////////
//                                         rep_pos_2d_sub_pxl.x = rep_list_id_data_i[i];
//                                         rep_pos_2d_sub_pxl.y = rep_list_id_data_j[j];
//                                         rep_sub_pixel_coord = conv_plan_pos_2_lin(rep_pos_2d_sub_pxl); // coord repeated frag
                                         ////////////////////////////////////////////
                                         norm_accu = int2float(list_accu_data_i[i] * list_accu_data_j[j]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         expected_trans = p.v_inter * norm_accu;
                                         // storage of inter sub frags contacts
//                                         full_exp_mat_rep[rep_sub_pixel_coord] = expected_trans;
                                         val_trans = local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]];
                                         local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = val_trans + expected_trans;
                                         if ( (loop_id_i == 0 ) && (loop_id_j == 0)){
                                             local_storage_id[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = sub_pixel_coord;
                                         }
        //                                 full_exp_mat[sub_pixel_coord] = tmp_val; // accumulation of sub frags
//                                         atomicAdd(&full_exp_mat[min(list_id_data_i[i], list_id_data_j[j]) * width_matrix +
//                                                                 max(list_id_data_i[i], list_id_data_j[j])],
//                                                                 (float) curand_poisson(&state[id_rng],expected_trans)); // accumulation of sub frags
                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             loop_id_j += 1;
                        }
                    }
                    loop_id_i += 1;
                }
            }

            for (i = 0; i <= init_limit_fi; i++){
                for(j = 0 * (on_diag == 0) + (i + 1) * (on_diag == 1);j <= init_limit_fj; j++){
                    val_expected = local_storage_exp[i][j];
                    full_exp_mat[local_storage_id[i][j]] = (float) curand_poisson(&state[id_rng], val_expected);
//                    full_exp_mat[local_storage_id[i][j]] = floor(val_expected);
                    val_pix = val_pix + val_expected;
                }
            }
            exp_mat[id_pix_2_write] = val_pix;
        }
    }


     __global__ void evaluate_likelihood(const float* __restrict obsData2D,
                                           frag* fragArray,
                                           int* collector_id,
                                           int2* dispatcher,
                                           int4* id_sub_frags,
                                           int4* rep_id_sub_frags,
                                           float3* len_bp_sub_frags,
                                           int3* accu_sub_frags,
                                           double* likelihood,
                                           param_simu* P,
                                           int max_id_up_diag,
                                           int max_id,
                                           int n_bins,
                                           int width_matrix,
                                           float n_frags_per_bins)
     {
        param_simu p = P[0];
        double val_likelihood = 0.0;
        double tmp_likelihood = 0.0;
        double val_observed = 0.0;
        double val_expected = 0.0;
        float s_tot, s, is, is0, is1, is2 = 0;
        float val_trans, val_cis = 0;
        float expected_cis  = 0.0f;
        float expected_trans  = 0.0f;
        float obsD = 0.0f;
        int contig_i, contig_j;
        int pos_i, pos_j, or_fi, or_fj, fi, fj, start_idx_i, start_idx_j, end_idx_i, end_idx_j;
        int2 pos_frag;
        float fi_start_bp, fj_start_bp;
        int3 accu_sub_frags_fi, accu_sub_frags_fj;
        int4 id_sub_frags_fi, id_sub_frags_fj;
        float3 len_bp_sub_frags_fi, len_bp_sub_frags_fj;
        float tmp_start_bp_fi, tmp_start_bp_fj;
        float len_sub_fi[3] = {0.0f,0.0f,0.0f};
        float len_sub_fj[3] = {0.0f,0.0f,0.0f};
        int idx_sub_fi[3] = {0,0,0};
        int idx_sub_fj[3] = {0,0,0};
        int rep_idx_sub_fi[3] = {0,0,0};
        int rep_idx_sub_fj[3] = {0,0,0};
        int accu_sub_fi[3] = {0,0,0};
        int accu_sub_fj[3] = {0,0,0};
        float list_s_fi[3] = {0.0f,0.0f,0.0f};
        float list_s_fj[3] = {0.0f,0.0f,0.0f};
        int list_id_data_i[3] = {0,0,0};
        int list_id_data_j[3] = {0,0,0};

        int list_accu_data_i[3] = {0,0,0};
        int list_accu_data_j[3] = {0,0,0};

        float local_storage_exp[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        float local_storage_obs[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        int local_sub_pos_fi[3] = {0,1,2};
        int local_sub_pos_fj[3] = {0,1,2};
        int id_data_fi = 0;
        int id_data_fj = 0;
        int o_id_data_fi = 0;
        int o_id_data_fj = 0;
        int tmp_id_data = 0;
        int init_limit_fi, init_limit_fj;
        int id_rep_fi, id_rep_fj;
        int2 data_pos, dispatch_fi, dispatch_fj, full_id_2d, pos_2d_sub_pxl , rep_pos_2d_sub_pxl;
        int sub_pixel_coord = 0;
        int swap = 0;
        int limit_fi = 0;
        int limit_fj = 0;
        float accu_i = 0.0f;
        float accu_j = 0.0f;
        int i,j;
        float norm_accu = 1.0;
        int id_pix_2_write, tmp_id_fi_fj, tmp_id_data_fi_fj;
        int id_pix_out = blockIdx.x * blockDim.x + threadIdx.x;
        float cumul_exp = 0.0f;
        int is_activ_fi, is_activ_fj;
        int are_the_same = 0;
        int is_rep_fi, is_rep_fj;
        int on_diag = 0;
        int loop_id_i = 0;
        int loop_id_j = 0;
        for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
            val_likelihood = 0.0;
            if (id_pix < max_id_up_diag){
                pos_frag = lin_2_2dpos(id_pix);
                o_id_data_fi = pos_frag.x; // frag_i absolute id
                o_id_data_fj = pos_frag.y; // frag_j absolute id
                on_diag = 0;
            }
            else{
                o_id_data_fi = (id_pix - max_id_up_diag) % n_bins;
                o_id_data_fj = o_id_data_fi;
                on_diag = 1;
            }

            dispatch_fi = dispatcher[o_id_data_fi];
            dispatch_fj = dispatcher[o_id_data_fj];
            // retrieve old likelihood
            // check if at least one of the two frags is repeated
            // create tmp_struct to store EXP[3,3]
            // retrieve OBS[3,3]
            // compute locL = P(OBS| EXP) for all of the nine values
            // S = locL - oldL
            ////// mandatory to check where are the sub frags !!!!! //////////////

            for (i=0;i<3;i++){
                for(j=0;j<3;j++){
                local_storage_exp[i][j] = 0.0f;
                }
            }
            init_limit_fi = id_sub_frags[fragArray->id_d[collector_id[dispatch_fi.x]]].w - 1;
            init_limit_fj = id_sub_frags[fragArray->id_d[collector_id[dispatch_fj.x]]].w - 1;
            ///// collect observed data ////
            loop_id_i = 0;
            loop_id_j = 0;
            for(id_rep_fi = dispatch_fi.x; id_rep_fi < dispatch_fi.y; id_rep_fi ++){
                fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                is_activ_fi = fragArray->activ[fi];
                is_rep_fi = fragArray->rep[fi];
                if (is_activ_fi == 1){
                    for(id_rep_fj = dispatch_fj.x; id_rep_fj< dispatch_fj.y; id_rep_fj ++){
                         fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                         fj = collector_id[id_rep_fj]; // index (with rep) frag curr_level
                         is_activ_fj = fragArray->activ[fj];
                         is_rep_fj = fragArray->rep[fj];
                         swap = 0;
                         are_the_same = (fi == fj);

                         if ((is_activ_fj == 1)){
//                         if ((is_activ_fj == 1) && ( (are_the_same == 0) || ((is_rep_fi == 0) && (is_rep_fj == 0)))){
                             norm_accu = 1.0f;

                             id_data_fi = fragArray->id_d[fi];
                             id_data_fj = fragArray->id_d[fj];

                             expected_cis  = 0.0f;
                             val_trans, val_cis = 0.0f;
                             contig_i = fragArray->id_c[fi];
                             contig_j = fragArray->id_c[fj];
                             if (contig_i == contig_j){
                                 pos_i = fragArray->pos[fi];
                                 pos_j = fragArray->pos[fj];
                                 if (pos_i > pos_j){ // fi is always the closest frag to the origin
                                    swap = 1;
                                    tmp_id_fi_fj = fi;
                                    fi = fj;
                                    fj = tmp_id_fi_fj;

                                    tmp_id_data_fi_fj = id_data_fi;
                                    id_data_fi = id_data_fj;
                                    id_data_fj = tmp_id_data_fi_fj;
                                 }
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];
                                 s_tot = int2float(fragArray->l_cont_bp[fi]) / 1000.0f;

                                 fi_start_bp =  int2float(fragArray->start_bp[fi]);
                                 fj_start_bp =  int2float(fragArray->start_bp[fj]);

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];

                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];

                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 len_sub_fi[0] = len_bp_sub_frags_fi.x;
                                 len_sub_fi[1] = len_bp_sub_frags_fi.y;
                                 len_sub_fi[2] = len_bp_sub_frags_fi.z;

                                 len_sub_fj[0] = len_bp_sub_frags_fj.x;
                                 len_sub_fj[1] = len_bp_sub_frags_fj.y;
                                 len_sub_fj[2] = len_bp_sub_frags_fj.z;

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;

                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[0];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[0] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[0];

                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[i];
                                        list_id_data_i[i] = idx_sub_fi[i];

                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[limit_fi];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[limit_fi] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];

                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[limit_fi - i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[limit_fi -i];
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];

                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi - i];
                                    }
                                 }
                                 if (or_fj == 1){
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[0];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[0] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[0];

                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[j];
                                        list_id_data_j[j] = idx_sub_fj[j];

                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[limit_fj];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[limit_fj] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];

                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[limit_fj - j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[limit_fj -j];
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];

                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i = 0; i <= limit_fi; i ++){
                                    for (j = 0 ; j <=  limit_fj; j ++){
                                         s = fabsf(list_s_fj[j] - list_s_fi[i]);
                                         norm_accu = int2float(list_accu_data_i[i] * list_accu_data_j[j]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         if (fragArray->circ[fi] == 1){
                                             expected_cis =  rippe_contacts_circ(s, s_tot, p) * norm_accu;
                                             //// DEBUG ///////////////////////////
//                                             expected_cis =  rippe_contacts_circ(s, s_tot, p);
                                         }
                                         else{
                                             expected_cis =  rippe_contacts(s, p) * norm_accu;
                                             //// DEBUG ///////////////////////////
//                                             expected_cis =  rippe_contacts(s, p);
                                         }
                                         // local storage of inter sub frags contacts
                                         if (swap == 0){
                                            val_cis = local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]];
                                            local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = val_cis + expected_cis;
                                             if ((loop_id_i == 0) && (loop_id_j == 0)){
                                                obsD = obsData2D[list_id_data_i[i] * width_matrix + list_id_data_j[j]];
                                                local_storage_obs[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = obsD;
                                             }

                                         }
                                         else{
                                            val_cis = local_storage_exp[local_sub_pos_fj[j]][local_sub_pos_fi[i]];
                                            local_storage_exp[local_sub_pos_fj[j]][local_sub_pos_fi[i]] = val_cis + expected_cis;
                                            if ((loop_id_i == 0) && (loop_id_j == 0)){
                                                obsD = obsData2D[list_id_data_i[i] * width_matrix + list_id_data_j[j]];
                                                local_storage_obs[local_sub_pos_fj[j]][local_sub_pos_fi[i]] = obsD;
                                            }
                                         }
                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             else{
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];

                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];
                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;
                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    list_id_data_i[0] = idx_sub_fi[0];

                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[i];

                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];

                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];

                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];

                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi];

                                    }
                                 }
                                 if (or_fj == 1){
                                    list_id_data_j[0] = idx_sub_fj[0];

                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[j];

                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];

                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];

                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i = 0; i <= limit_fi; i ++){
                                    for (j = 0 ; j <=  limit_fj; j ++){

                                         norm_accu = int2float(list_accu_data_i[i] * list_accu_data_j[j]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         expected_trans = p.v_inter * norm_accu;
                                         //// DEBUG ///////////////////////////
//                                         expected_trans = p.v_inter;
                                         val_trans = local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]];
                                         local_storage_exp[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = val_trans + expected_trans;
                                         if ((loop_id_i == 0) && (loop_id_j == 0)){
                                            obsD = obsData2D[list_id_data_i[i] * width_matrix + list_id_data_j[j]];
                                            local_storage_obs[local_sub_pos_fi[i]][local_sub_pos_fj[j]] = obsD;
                                         }

                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             loop_id_j += 1;
                        }
                    }
                    loop_id_i += 1;
                }
            }

            for (i = 0; i <= init_limit_fi; i++){
                for(j = 0 * (on_diag == 0) + (i + 1) * (on_diag == 1);j <= init_limit_fj; j++){
                    val_expected = (double) local_storage_exp[i][j];
                    val_observed = (double) local_storage_obs[i][j];
                    tmp_likelihood = evaluate_likelihood_double(val_expected, val_observed) + val_likelihood;
                    val_likelihood = tmp_likelihood;
                }
            }
            likelihood[id_pix] = val_likelihood;
        }
    }


    __global__ void fill_sub_index_fA(frag* fragArray, int* sub_index, int contig_A, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){
            int contig_fi = fragArray->id_c[id_frag];
            if (contig_fi == contig_A){
                int pos_i = fragArray->pos[id_frag];
                int id_d = fragArray->id_d[id_frag];
                sub_index[pos_i] = id_d;
            }
        }
    }

    __global__ void fill_sub_index_fB(frag* fragArray, int* sub_index, int contig_B, int l_cont_fA, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){
            int contig_fi = fragArray->id_c[id_frag];
            if (contig_fi == contig_B){
                int pos_i = fragArray->pos[id_frag];
                int id_d = fragArray->id_d[id_frag];
                sub_index[l_cont_fA + pos_i] = id_d;
            }
        }
    }

    __global__ void set_null(float* vect, int max_id)
    {
        int id_pix = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_pix < max_id){
            vect[id_pix] = 0.0;
        }
    }

    __global__ void sub_compute_likelihood(const float* __restrict obsData2D,
                                           frag* fragArray,
                                           int* sub_index,
                                           int* list_rep_frag,
                                           int* list_uniq_frag,
                                           int* collector_id,
                                           int2* dispatcher,
                                           int4* id_sub_frags,
                                           float3* len_bp_sub_frags,
                                           int3* accu_sub_frags,
                                           double* likelihood,
                                           double* curr_likelihood,
                                           param_simu* P,
                                           int max_id_no_repeats,
                                           int lim_repeats_vs_uniq,
                                           int lim_intra_repeats,
                                           int max_id,
                                           int n_frags_uniq,
                                           int n_repeats,
                                           int width_matrix,
                                           int n_bins,
                                           float n_frags_per_bins)
    {
        extern __shared__ double res[];
        param_simu p = P[0];
        double val_out = 0.0;
        double tmp_likelihood = 0.0;
        float obsD;
        float s_tot, s, is, is0, is1, is2 = 0;
        float val_trans, val_cis = 0;
        float expected_cis  = 0;
        float expected_trans  = 0;
        int contig_i, contig_j;
        int pos_i, pos_j, or_fi, or_fj, fi, fj;
        int2 pos_frag;
        float fi_start_bp, fj_start_bp;
        int3 accu_sub_frags_fi, accu_sub_frags_fj;
        int4 id_sub_frags_fi, id_sub_frags_fj;
        float3 len_bp_sub_frags_fi, len_bp_sub_frags_fj;
        float tmp_start_bp_fi, tmp_start_bp_fj;
        float len_sub_fi[3] = {0.0f, 0.0f, 0.0f};
        float len_sub_fj[3] = {0.0f, 0.0f, 0.0f};
        int idx_sub_fi[3] = {0, 0, 0};
        int idx_sub_fj[3] = {0, 0, 0};
        int accu_sub_fi[3] = {0,0,0};
        int accu_sub_fj[3] = {0,0,0};
        int id_pix_out = blockIdx.x * blockDim.x + threadIdx.x;
        double tmp_val = 0.0;
        double val_curr_id_pix = 0;
        int2 glob_pos;
        float list_s_fi[3] = {0.0f,0.0f,0.0f};
        float list_s_fj[3] = {0.0f,0.0f,0.0f};
        int list_id_data_i[3] = {0,0,0};
        int list_id_data_j[3] = {0,0,0};
        int list_accu_data_i[3] = {0,0,0};
        int list_accu_data_j[3] = {0,0,0};
        int limit_fi = 0;
        int limit_fj = 0;
        float accu_i = 0.0f;
        float accu_j = 0.0f;
        int i,j;
        int glob_index;

        float norm_accu = 1.0;
        float local_storage_exp[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        float local_storage_obs[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
//        int local_storage_id[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        int local_sub_pos_fi[3] = {0,1,2};
        int local_sub_pos_fj[3] = {0,1,2};
        int id_data_fi = 0;
        int id_data_fj = 0;
        int o_id_data_fi = 0;
        int o_id_data_fj = 0;
        int tmp_id_data = 0;
        int init_limit_fi, init_limit_fj, tmp_id_fi_fj, tmp_id_data_fi_fj;;
        int id_rep_fi, id_rep_fj;
        int2 data_pos, dispatch_fi, dispatch_fj, full_id_2d, pos_2d_sub_pxl;
        int sub_pixel_coord = 0;
//        int rep_sub_pixel_coord = 0;
        int swap = 0;
        double old_likelihood = 0.0;
        double val_likelihood = 0.0;
        double val_observed = 0.0;
        double val_expected = 0.0;
        int is_activ_fi, is_activ_fj;
        int are_the_same = 0;
        int is_rep_fi, is_rep_fj;;
        int on_diag = 0;
        int tmp_id_data_fi, tmp_id_data_fj;
        int loop_id_i = 0;
        int loop_id_j = 0;
        int i_spec, j_spec;
        for (int id_pix = blockIdx.x * blockDim.x + threadIdx.x; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
            val_likelihood = 0.0;
//            o_id_data_fi = 0;
//            o_id_data_fj = 0;
            on_diag = 0;
            if (id_pix < max_id_no_repeats){
                pos_frag = lin_2_2dpos(id_pix);
                tmp_id_data_fi = sub_index[pos_frag.x]; // frag_i absolute id
                tmp_id_data_fj = sub_index[pos_frag.y]; // frag_j absolute id
                o_id_data_fi = min(tmp_id_data_fi, tmp_id_data_fj);
                o_id_data_fj = max(tmp_id_data_fi, tmp_id_data_fj);
            }
            else if((id_pix >= max_id_no_repeats) && (id_pix < lim_repeats_vs_uniq)){
                tmp_id_data_fi = list_rep_frag[(id_pix - max_id_no_repeats) / n_frags_uniq];
                tmp_id_data_fj = list_uniq_frag[(id_pix - max_id_no_repeats) % n_frags_uniq];
                o_id_data_fi = min(tmp_id_data_fi, tmp_id_data_fj);
                o_id_data_fj = max(tmp_id_data_fi, tmp_id_data_fj);
            }
            else if ((id_pix >= lim_repeats_vs_uniq) && (id_pix < lim_intra_repeats)){
                pos_frag = lin_2_2dpos(id_pix - lim_repeats_vs_uniq);
                tmp_id_data_fi = list_rep_frag[pos_frag.x]; // frag_i absolute id
                tmp_id_data_fj = list_rep_frag[pos_frag.y]; // frag_j absolute id
                o_id_data_fi = min(tmp_id_data_fi, tmp_id_data_fj);
                o_id_data_fj = max(tmp_id_data_fi, tmp_id_data_fj);
            }
            else{
                o_id_data_fi = list_rep_frag[(id_pix - lim_intra_repeats) % n_repeats];
                o_id_data_fj = o_id_data_fi;
                on_diag = 1;
            }
//            on_diag = (o_id_data_fi == o_id_data_fj);

            dispatch_fi = dispatcher[o_id_data_fi];
            dispatch_fj = dispatcher[o_id_data_fj];

            ///// collect old likelihod ////

            if (on_diag == 0){
                glob_pos.x = o_id_data_fi;
                glob_pos.y = o_id_data_fj;
                glob_index = conv_plan_pos_2_lin(glob_pos);
            }
            else{
                glob_index = (n_bins * (n_bins - 1) / 2) + o_id_data_fi;
            }

            old_likelihood = curr_likelihood[glob_index];

            for (i=0;i < 3;i++){ // init local storage
                for(j=0;j < 3;j++){
                local_storage_exp[i][j] = 0.0f;
                }
            }
            init_limit_fi = id_sub_frags[fragArray->id_d[collector_id[dispatch_fi.x]]].w - 1;
            init_limit_fj = id_sub_frags[fragArray->id_d[collector_id[dispatch_fj.x]]].w - 1;
            loop_id_i = 0;
            loop_id_j = 0;
            for(id_rep_fi = dispatch_fi.x; id_rep_fi< dispatch_fi.y; id_rep_fi ++){
                fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                is_activ_fi = fragArray->activ[fi];
                is_rep_fi = fragArray->rep[fi];
                if (is_activ_fi == 1){
                    for(id_rep_fj = dispatch_fj.x; id_rep_fj< dispatch_fj.y; id_rep_fj ++){
                         fi = collector_id[id_rep_fi]; // index (with rep) frag curr_level
                         fj = collector_id[id_rep_fj]; // index (with rep) frag curr_level
                         is_activ_fj = fragArray->activ[fj];
                         is_rep_fj = fragArray->rep[fj];
                         are_the_same = (fi == fj);
                         swap = 0;
                         if ((is_activ_fj == 1)){
//                         if ((is_activ_fj == 1) && ( are_the_same == 0)){
                             norm_accu = 1.0f;

                             id_data_fi = fragArray->id_d[fi];
                             id_data_fj = fragArray->id_d[fj];
                             data_pos.x = id_data_fi;
                             data_pos.y = id_data_fj;

                             expected_cis  = 0.0f;
                             val_trans, val_cis = 0.0f;
                             contig_i = fragArray->id_c[fi];
                             contig_j = fragArray->id_c[fj];
                             if (contig_i == contig_j){
                                 pos_i = fragArray->pos[fi];
                                 pos_j = fragArray->pos[fj];
                                 if (pos_i > pos_j){ // fi is always the closest frag to the origin
                                    swap = 1;
                                    tmp_id_fi_fj = fi;
                                    fi = fj;
                                    fj = tmp_id_fi_fj;

                                    tmp_id_data_fi_fj = id_data_fi;
                                    id_data_fi = id_data_fj;
                                    id_data_fj = tmp_id_data_fi_fj;
                                 }
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];
                                 s_tot = int2float(fragArray->l_cont_bp[fi]) / 1000.0f;

                                 fi_start_bp =  int2float(fragArray->start_bp[fi]);
                                 fj_start_bp =  int2float(fragArray->start_bp[fj]);

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];
                                 /////////////////////////////////
                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];

                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 len_sub_fi[0] = len_bp_sub_frags_fi.x;
                                 len_sub_fi[1] = len_bp_sub_frags_fi.y;
                                 len_sub_fi[2] = len_bp_sub_frags_fi.z;

                                 len_sub_fj[0] = len_bp_sub_frags_fj.x;
                                 len_sub_fj[1] = len_bp_sub_frags_fj.y;
                                 len_sub_fj[2] = len_bp_sub_frags_fj.z;

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;

                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[0];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[0] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[0];
                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[i];
                                        list_id_data_i[i] = idx_sub_fi[i];
                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    accu_i = fi_start_bp/1000.0f + len_sub_fi[limit_fi];
                                    list_s_fi[0] = fi_start_bp / 1000.0f + len_sub_fi[limit_fi] / 2.0f;
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];
                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_s_fi[i] = accu_i + len_sub_fi[limit_fi - i] / 2.0f;
                                        accu_i = accu_i + len_sub_fi[limit_fi -i];
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi - i];
                                    }
                                 }
                                 if (or_fj == 1){
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[0];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[0] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[0];
                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[j];
                                        list_id_data_j[j] = idx_sub_fj[j];
                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    accu_j = fj_start_bp/1000.0f + len_sub_fj[limit_fj];
                                    list_s_fj[0] = fj_start_bp / 1000.0f + len_sub_fj[limit_fj] / 2.0f;
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];
                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_s_fj[j] = accu_j + len_sub_fj[limit_fj - j] / 2.0f;
                                        accu_j = accu_j + len_sub_fj[limit_fj -j];
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i_spec = 0; i_spec <= limit_fi; i_spec ++){
                                    for (j_spec = 0; j_spec <=  limit_fj; j_spec ++){
                                         s = fabsf(list_s_fj[j_spec] - list_s_fi[i_spec]);
                                         ////////////////////////////////////////////
                                         norm_accu = int2float(list_accu_data_i[i_spec] * list_accu_data_j[j_spec]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         if (fragArray->circ[fi] == 1){
                                             expected_cis =  rippe_contacts_circ(s, s_tot, p) * norm_accu;
                                             //// DEBUG ///////////////////////////
//                                             expected_cis =  rippe_contacts_circ(s, s_tot, p);
                                         }
                                         else{
                                             expected_cis =  rippe_contacts(s, p) * norm_accu;
                                             //// DEBUG ///////////////////////////
//                                             expected_cis =  rippe_contacts(s, p);
                                         }
                                         // local storage of inter sub frags contacts
                                         if (swap == 0){
                                            val_cis = local_storage_exp[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] + expected_cis;
                                            local_storage_exp[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] = val_cis;
                                             if ((loop_id_i == 0) && (loop_id_j == 0)){
                                                 obsD = obsData2D[list_id_data_i[i_spec] * width_matrix + list_id_data_j[j_spec]];
                                                 local_storage_obs[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] = obsD;
                                             }
                                         }
                                         else{
                                            val_cis = local_storage_exp[local_sub_pos_fj[j_spec]][local_sub_pos_fi[i_spec]] + expected_cis;
                                            local_storage_exp[local_sub_pos_fj[j_spec]][local_sub_pos_fi[i_spec]] = val_cis;
                                            if ((loop_id_i == 0) && (loop_id_j == 0)){
                                                 obsD = obsData2D[list_id_data_j[j_spec] * width_matrix + list_id_data_i[i_spec]];
                                                 local_storage_obs[local_sub_pos_fj[j_spec]][local_sub_pos_fi[i_spec]] = obsD;
                                            }
                                         }
                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             else{
                                 or_fi = fragArray->ori[fi];
                                 or_fj = fragArray->ori[fj];

                                 id_sub_frags_fi = id_sub_frags[id_data_fi];
                                 id_sub_frags_fj = id_sub_frags[id_data_fj];
                                 /////////////////////////////////
                                 len_bp_sub_frags_fi = len_bp_sub_frags[id_data_fi];
                                 len_bp_sub_frags_fj = len_bp_sub_frags[id_data_fj];
                                 accu_sub_frags_fi = accu_sub_frags[id_data_fi];
                                 accu_sub_frags_fj = accu_sub_frags[id_data_fj];

                                 idx_sub_fi[0] = id_sub_frags_fi.x;
                                 idx_sub_fi[1] = id_sub_frags_fi.y;
                                 idx_sub_fi[2] = id_sub_frags_fi.z;

                                 idx_sub_fj[0] = id_sub_frags_fj.x;
                                 idx_sub_fj[1] = id_sub_frags_fj.y;
                                 idx_sub_fj[2] = id_sub_frags_fj.z;
                                 //////////////////////////////////
                                 accu_sub_fi[0] = accu_sub_frags_fi.x;
                                 accu_sub_fi[1] = accu_sub_frags_fi.y;
                                 accu_sub_fi[2] = accu_sub_frags_fi.z;

                                 accu_sub_fj[0] = accu_sub_frags_fj.x;
                                 accu_sub_fj[1] = accu_sub_frags_fj.y;
                                 accu_sub_fj[2] = accu_sub_frags_fj.z;


                                ////////////// under construction ///////////////////
                                 limit_fi = id_sub_frags_fi.w - 1;
                                 limit_fj = id_sub_frags_fj.w - 1;

                                 if (or_fi == 1){
                                    list_id_data_i[0] = idx_sub_fi[0];
                                    local_sub_pos_fi[0] = 0;
                                    list_accu_data_i[0] = accu_sub_fi[0];
                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[i];
                                        local_sub_pos_fi[i] = i;
                                        list_accu_data_i[i] = accu_sub_fi[i];
                                    }
                                 }
                                 else{
                                    list_id_data_i[0] = idx_sub_fi[limit_fi];
                                    local_sub_pos_fi[0] = limit_fi;
                                    list_accu_data_i[0] = accu_sub_fi[limit_fi];

                                    for (i = 1; i <= limit_fi; i++){
                                        list_id_data_i[i] = idx_sub_fi[limit_fi - i];
                                        local_sub_pos_fi[i] = limit_fi - i;
                                        list_accu_data_i[i] = accu_sub_fi[limit_fi];

                                    }
                                 }
                                 if (or_fj == 1){
                                    list_id_data_j[0] = idx_sub_fj[0];
                                    local_sub_pos_fj[0] = 0;
                                    list_accu_data_j[0] = accu_sub_fj[0];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[j];
                                        local_sub_pos_fj[j] = j;
                                        list_accu_data_j[j] = accu_sub_fj[j];
                                    }
                                 }
                                 else{
                                    list_id_data_j[0] = idx_sub_fj[limit_fj];
                                    local_sub_pos_fj[0] = limit_fj;
                                    list_accu_data_j[0] = accu_sub_fj[limit_fj];
                                    for (j = 1; j <= limit_fj; j++){
                                        list_id_data_j[j] = idx_sub_fj[limit_fj - j];
                                        local_sub_pos_fj[j] = limit_fj - j;
                                        list_accu_data_j[j] = accu_sub_fj[limit_fj - j];
                                    }
                                 }

                                 for (i_spec = 0; i_spec <= limit_fi; i_spec ++){
                                    for (j_spec = 0; j_spec <= limit_fj; j_spec ++){
                                         ////////////////////////////////////////////
                                         if ((loop_id_i == 0) && (loop_id_j == 0)){
                                             obsD = obsData2D[list_id_data_i[i_spec] * width_matrix + list_id_data_j[j_spec]];
                                             local_storage_obs[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] = obsD;
                                         }
                                         ////////////////////////////////////////////
                                         norm_accu = int2float(list_accu_data_i[i_spec] * list_accu_data_j[j_spec]) / n_frags_per_bins;
                                         /// DEBUG ////
//                                         norm_accu = 1;
                                         /// DEBUG ////
                                         expected_trans = p.v_inter * norm_accu;
                                         /////////// DEBUG ///////////////////////////////////
//                                         expected_trans = p.v_inter;
                                         // storage of inter sub frags contacts
                                         val_trans = local_storage_exp[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] + expected_trans;
                                         local_storage_exp[local_sub_pos_fi[i_spec]][local_sub_pos_fj[j_spec]] = val_trans;

                                    }
                                 }
                                ////////////// end construction ///////////////////
                             }
                             loop_id_j += 1;
                        }
                    }
                    loop_id_i += 1;
                }
            }
            tmp_likelihood = 0.0;
            for (i=0;i<=init_limit_fi;i++){
                for(j = 0 * (on_diag == 0) + (i + 1) * (on_diag == 1); j <= init_limit_fj; j++){
                    val_expected = (double) local_storage_exp[i][j];
                    val_observed = (double) local_storage_obs[i][j];
                    tmp_likelihood = val_likelihood + evaluate_likelihood_double(val_expected, val_observed);
                    val_likelihood = tmp_likelihood;
                }
            }

//            atomicAdd(&likelihood[0], val_likelihood - old_likelihood);
            tmp_likelihood = val_out + val_likelihood - old_likelihood;
            val_out = tmp_likelihood;

        }

        res[threadIdx.x] = val_out ;
        __syncthreads();
        val_out = 0.0;
        if (threadIdx.x == 0){
            for (i = 0; i < blockDim.x; i++){
                tmp_likelihood = val_out + res[i];
                val_out = tmp_likelihood;
            }
            atomicAdd(&likelihood[0], val_out);
        }
    }

    __global__ void copy_struct(frag* fragArray, frag* smplfragArray, int* id_contigs, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        int id_c = 0;
        if (id_frag  < n_frags){
            fragArray->pos[id_frag] = smplfragArray->pos[id_frag];
            id_c = smplfragArray->id_c[id_frag];
            fragArray->id_c[id_frag] = id_c;
            id_contigs[id_frag] = id_c;
            fragArray->circ[id_frag] = smplfragArray->circ[id_frag];
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = smplfragArray->ori[id_frag];
            fragArray->start_bp[id_frag] = smplfragArray->start_bp[id_frag];
            fragArray->len_bp[id_frag] = smplfragArray->len_bp[id_frag];
            fragArray->prev[id_frag] = smplfragArray->prev[id_frag];
            fragArray->next[id_frag] = smplfragArray->next[id_frag];
            fragArray->l_cont[id_frag] = smplfragArray->l_cont[id_frag];
            fragArray->l_cont_bp[id_frag] = smplfragArray->l_cont_bp[id_frag];
            fragArray->rep[id_frag] = smplfragArray->rep[id_frag];
            fragArray->activ[id_frag] = smplfragArray->activ[id_frag];
            fragArray->id_d[id_frag] = smplfragArray->id_d[id_frag];
        }
    }

    __global__ void copy_gpu_array(double* dest, double* input, int max_id)
    {
        int id_pix_out = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_pix_out < max_id){
            for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
                dest[id_pix] = input[id_pix];
            }
        }
    }


    __global__ void simple_copy(frag* fragArray, frag* smplfragArray, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag  < n_frags){
            fragArray->pos[id_frag] = smplfragArray->pos[id_frag];
            fragArray->id_c[id_frag] = smplfragArray->id_c[id_frag];
            fragArray->circ[id_frag] = smplfragArray->circ[id_frag];
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = smplfragArray->ori[id_frag];
            fragArray->start_bp[id_frag] = smplfragArray->start_bp[id_frag];
            fragArray->len_bp[id_frag] = smplfragArray->len_bp[id_frag];
            fragArray->prev[id_frag] = smplfragArray->prev[id_frag];
            fragArray->next[id_frag] = smplfragArray->next[id_frag];
            fragArray->l_cont[id_frag] = smplfragArray->l_cont[id_frag];
            fragArray->l_cont_bp[id_frag] = smplfragArray->l_cont_bp[id_frag];
            fragArray->rep[id_frag] = smplfragArray->rep[id_frag];
            fragArray->activ[id_frag] = smplfragArray->activ[id_frag];
            fragArray->id_d[id_frag] = smplfragArray->id_d[id_frag];
        }
    }


    __global__ void reorder_tex(unsigned char* data, int* index_new, int n_frags)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        if ((i < n_frags) && (j < n_frags))
        {
            int i_new = index_new[i];
            int j_new = index_new[j];

            unsigned char out = tex2D(tex, (float) i_new, (float) j_new);
            data[i + j * n_frags]  = out;

        }
    }

//    __global__ void reorder_tex(unsigned char* data, frag* fragArray, int* cum_id_c, int n_frags)
//    {
//        int i = threadIdx.x + blockDim.x * blockIdx.x;
//        int j = threadIdx.y + blockDim.y * blockIdx.y;
//        if ((i < n_frags) && (j < n_frags))
//        {
//            int rep_i = fragArray->rep[i];
//            int rep_j = fragArray->rep[j];
//            if ( (rep_i == 0) && (rep_j == 0))
//            {
//                int pos_i = fragArray->pos[i];
//                int pos_j = fragArray->pos[j];
//
//                int id_c_i = fragArray->id_c[i];
//                int id_c_j = fragArray->id_c[j];
//
//                int offset_i = cum_id_c[id_c_i];
//                int offset_j = cum_id_c[id_c_j];
//
//                int i_new = offset_i + pos_i;
//                int j_new = offset_j + pos_j;
//
//                unsigned char out = tex2D(tex, (float) i, (float) j);
////                data[i_new + j_new * n_frags]  = out;
//            }
//        }
//    }





    __global__ void gl_update_pos(float4* pos,
                                  float4* color,
                                  float4* vel,
                                  float4* pos_gen,
                                  float4* vel_gen,
                                  frag* fragArray,
                                  int * old_2_new_idx,
                                  int* id_contigs,
                                  float max_len,
                                  float max_id,
                                  int n_frags,
                                  int id_fi,
                                  int min_id_c_new,
                                  curandState* state, int n_rng, float dt)
    {
        //get our index in the array
        int id_frag =  threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag  < n_frags){

            int id_rng = id_frag % n_rng;
            float shift_y = (curand_normal(&state[id_rng]))*0.01;
            float shift_x = (curand_normal(&state[id_rng]))*0.01;
            float shift_z = (curand_normal(&state[id_rng]))*0.01;
            float shift_rot = shift_y;
            int id_c = fragArray->id_c[id_frag];
            int id_c_new = old_2_new_idx[id_c];
            fragArray->id_c[id_frag] = id_c_new;
            id_contigs[id_frag] = id_c_new;
            int is_circ = fragArray->circ[id_frag];
//            int is_circ = id_c_new % 2 == 0;
            float life = vel[id_frag].w;
            float pos_x;
            float l_cont;
            float radius;
            if (fragArray->l_cont[id_frag] > 1){

                pos_x = (int2float(fragArray->pos[id_frag]))/ max_len;
                if (is_circ == 1){
                    radius = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new) / 2 ;
                    l_cont = int2float(fragArray->l_cont[id_frag]) / max_len + 0.01f;
                    pos[id_frag].x = radius * 2;
                    pos[id_frag].y = 0 + radius * cos((pos_x + shift_rot) * 2 * M_PI  / l_cont ); // x plan coord
                    pos[id_frag].z = 0 + radius * sin((pos_x + shift_rot) * 2 * M_PI / l_cont ); // y plan coord;

                }
                else{
                    pos[id_frag].x = pos_x;
                    pos[id_frag].y = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new);
    //                pos[id_frag].x = int2float(fragArray->pos[id_frag])/ max_len + 0.01f;
    //                pos[id_frag].y = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new) + 0.01f;
                    pos[id_frag].z = 0;

                }
                color[id_frag].w = 1.5;
            }
            else{
                float4 p = pos[id_frag];
                float4 v = vel[id_frag];
                life -= dt;
                if(life <= 0.f)
                {
                    p = pos_gen[id_frag];
                    v = vel_gen[id_frag];
                    life = 1.0f;
                }
                v.z -= 9.8f*dt;

                p.x += shift_y;;
                p.y += shift_z;
                p.z += shift_x;
                v.w = life;

                //update the arrays with our newly computed values
                pos[id_frag] = p;
                vel[id_frag] = v;

                //you can manipulate the color based on properties of the system
                //here we adjust the alpha
                color[id_frag].w = life;
            }
        }
    }




//    __global__ void gl_update_pos(float4* pos,
//                                  float4* color,
//                                  float4* vel,
//                                  float4* pos_gen,
//                                  float4* vel_gen,
//                                  frag* fragArray,
//                                  int * old_2_new_idx,
//                                  int* id_contigs,
//                                  float max_len,
//                                  float max_id,
//                                  int n_frags,
//                                  int id_fi,
//                                  int min_id_c_new,
//                                  curandState* state, int n_rng, float dt)
//    {
//        //get our index in the array
//        int id_frag =  threadIdx.x + blockDim.x * blockIdx.x;
//        if (id_frag  < n_frags){
//
//            int id_rng = id_frag % n_rng;
//            float shift_y = (curand_normal(&state[id_rng]) * 0.25) + 0.25;
//            int id_c = fragArray->id_c[id_frag];
//            int id_c_new = old_2_new_idx[id_c];
//            fragArray->id_c[id_frag] = id_c_new;
//            id_contigs[id_frag] = id_c_new;
//            float life = vel[id_frag].w;
//            if (fragArray->l_cont[id_frag] > 1){
//                pos[id_frag].x = int2float(fragArray->pos[id_frag])/ max_len + 0.01f;
//                pos[id_frag].y = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new);
////                pos[id_frag].x = int2float(fragArray->pos[id_frag])/ max_len + 0.01f;
////                pos[id_frag].y = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new) + 0.01f;
//                pos[id_frag].z = 0;
//                color[id_frag].w = 1.5;
//            }
//            else{
//                float4 p = pos[id_frag];
//                float4 v = vel[id_frag];
//                life -= dt;
//                if(life <= 0.f)
//                {
//                    p = pos_gen[id_frag];
//                    v = vel_gen[id_frag];
//                    life = 1.0f;
//                }
//                v.z -= 9.8f*dt;
//
//                p.x += v.x*dt;
//                p.y += v.y*dt;
//                p.z += v.z*dt;
//                v.w = life;
//
//                //update the arrays with our newly computed values
//                pos[id_frag] = p;
//                vel[id_frag] = v;
//
//                //you can manipulate the color based on properties of the system
//                //here we adjust the alpha
//                color[id_frag].w = life;
//            }
//        }
//    }

} // extern "C"


