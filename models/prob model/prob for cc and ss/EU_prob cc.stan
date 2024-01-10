
functions {
  real pweight(real p_val, real gamma) {
    return exp(-pow(-log(p_val), gamma));
  }
}



data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of participants
	int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=-1,upper=1> cho[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt
  real oa[N,2];
  real ob[N,2];
  real pa[N,2];
  real pb[N,2];
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
	int<lower=-1,upper=1> con[N];  // condition index, -1 = simple vs. simple, 1 = complex vs. complex
}



parameters {
	real mu_alpha;
	real mu_theta;
	real mu_threshold;
	real mu_ndt;
	real mu_gamma;
	real mu_delta_gamma;

	real<lower=0> sd_alpha;
	real<lower=0> sd_theta;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;
	real<lower=0> sd_gamma;
	real<lower=0> sd_delta_gamma;
	
	
	real z_theta[L];
	real z_threshold[L];
	real z_alpha[L];
	real z_ndt[L];
	real z_gamma[L];
	real z_delta_gamma[L];
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real ua[N];	
	real ub[N];	
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt
	real<lower=0> sum_delta_gamma[N];



	real<lower=0> alpha_sbj[L];
	real<lower=0> theta_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];
	real<lower=0> gamma_sbj[L];
	real<lower=0> delta_gamma_sbj[L];


	real transf_mu_alpha;
	real transf_mu_theta;
	real transf_mu_threshold;
	real transf_mu_ndt;
	real transf_mu_gamma;
	real transf_mu_delta_gamma;


	transf_mu_alpha = log(1+ exp(mu_alpha));						// for the output
	transf_mu_theta = log(1+ exp(mu_theta));					
	transf_mu_threshold = log(1+ exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_gamma = log(1 + exp(mu_gamma));
	transf_mu_delta_gamma = mu_delta_gamma;


	for (l in 1:L) {
	  alpha_sbj[l] = log(1 + exp(mu_alpha + z_alpha[l]*sd_alpha));
		theta_sbj[l] = log(1 + exp(mu_theta + z_theta[l]*sd_theta));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
		gamma_sbj[l] = mu_gamma + z_gamma[l]*sd_gamma;
		delta_gamma_sbj[l] = mu_delta_gamma + z_delta_gamma[l]*sd_delta_gamma;
	}

	for (n in 1:N) {
	  sum_delta_gamma[n] = log(1+exp(gamma_sbj[participant[n]] + delta_gamma_sbj[participant[n]]*con[n]));
		ua[n] = pweight(pa[n,1], sum_delta_gamma[n]) * pow(oa[n,1],alpha_sbj[participant[n]]) + (1- pweight(pa[n,1], sum_delta_gamma[n])) * pow(oa[n,2],alpha_sbj[participant[n]]);
		ub[n] = pweight(pb[n,1], sum_delta_gamma[n]) * pow(ob[n,1],alpha_sbj[participant[n]]) + (1- pweight(pb[n,1], sum_delta_gamma[n])) * pow(ob[n,2],alpha_sbj[participant[n]]);
		drift_t[n] = theta_sbj[participant[n]] * (ua[n] - ub[n]);
		drift_ll[n] = drift_t[n]*cho[n];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];
	}
}
model {
	mu_alpha ~ normal(0.5413, 2);
	mu_theta ~ normal(1, 5);
	mu_threshold ~ normal(1, 3);
	mu_ndt ~ normal(0, 1);
	mu_gamma ~ normal(0, 5);
	mu_delta_gamma ~ normal(0, 5);

	sd_alpha ~ normal(0, 2);
	sd_theta ~ normal(0, 5);
	sd_threshold ~ normal(0,3);
	sd_ndt ~ normal(0,1);
	sd_gamma ~ normal(0, 5);
	sd_delta_gamma ~ normal(0, 5);
	
	
	z_alpha ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);
	z_theta ~ normal(0, 1);
	z_gamma ~ normal(0, 1);
	z_delta_gamma ~ normal(0, 1);

	

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
}
}

