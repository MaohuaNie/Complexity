data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of participants
	int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=-1,upper=1> cho[N];				// complex choice 1, easy choice -1
	real<lower=0> rt[N];							// rt
	
	real evd[N];
	real sdd[N];
	real trialtype1[N];          // 1 rl -1 lr 0 ns
	real trialtype2[N];  
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
	int<lower=-1,upper=1> con[N];  // condition index, -1 = simple vs. simple, 1 = complex vs. complex

}
parameters {
	real mu_alpha;
	real mu_theta;
	real mu_threshold;
	real mu_ndt;
	real mu_delta1;
	real mu_delta2;
	real mu_zeta1;
	real mu_zeta2;

	

	real<lower=0> sd_alpha;
	real<lower=0> sd_theta;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;
	real<lower=0> sd_delta1;
	real<lower=0> sd_delta2;
	real<lower=0> sd_zeta1;
	real<lower=0> sd_zeta2;

	
	
	real z_theta[L];
	real z_threshold[L];
	real z_alpha[L];
	real z_ndt[L];
	real z_delta1[L];
	real z_delta2[L];
	real z_zeta1[L];
	real z_zeta2[L];


}

transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt






	real alpha_sbj[L];
	real<lower=0> theta_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];
	real delta1_sbj[L];
	real delta2_sbj[L];
	real zeta1_sbj[L];
	real zeta2_sbj[L];



	real transf_mu_alpha;
	real transf_mu_theta;
	real transf_mu_threshold;
	real transf_mu_ndt;
	real transf_mu_delta1;
	real transf_mu_delta2;
	real transf_mu_zeta1;
	real transf_mu_zeta2;



	transf_mu_alpha = mu_alpha;						// for the output
	transf_mu_theta = log(1+ exp(mu_theta));					
	transf_mu_threshold = log(1+ exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_delta1 = mu_delta1;
	transf_mu_delta2 = mu_delta2;
	transf_mu_zeta1 = mu_zeta1;
	transf_mu_zeta2 = mu_zeta2;



	for (l in 1:L) {
		alpha_sbj[l] = mu_alpha + z_alpha[l]*sd_alpha;
		theta_sbj[l] = log(1 + exp(mu_theta + z_theta[l]*sd_theta));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
		delta1_sbj[l] = mu_delta1 + z_delta1[l]*sd_delta1;
		delta2_sbj[l] = mu_delta2 + z_delta2[l]*sd_delta2;
		zeta1_sbj[l] = mu_zeta1 + z_zeta1[l]*sd_zeta1;
		zeta2_sbj[l] = mu_zeta2 + z_zeta2[l]*sd_zeta2;
  

	}

	for (n in 1:N) {
		drift_t[n] = theta_sbj[participant[n]] *  (evd[n] + alpha_sbj[participant[n]] * sdd[n] + ((delta1_sbj[participant[n]] + zeta1_sbj[participant[n]] *con[n]) * trialtype1[n] + (delta2_sbj[participant[n]] + zeta2_sbj[participant[n]] *con[n])* trialtype2[n]));
		drift_ll[n] = drift_t[n]*cho[n];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];


	}
}
model {
	mu_alpha ~ normal(0, 5);
	mu_theta ~ normal(1, 5);
	mu_threshold ~ normal(1, 3);
	mu_ndt ~ normal(0, 1);
	mu_delta1 ~ normal(0, 5);
	mu_delta2 ~ normal(0, 5);
	mu_zeta1 ~ normal(0, 5);
	mu_zeta2 ~ normal(0, 5);


	sd_alpha ~ normal(0, 5);
	sd_theta ~ normal(0, 5);
	sd_threshold ~ normal(0,3);
	sd_ndt ~ normal(0,1);
	sd_delta1 ~ normal(0,5);
	sd_delta2 ~ normal(0,5);
	sd_zeta1 ~ normal(0,5);
	sd_zeta2 ~ normal(0,5);

	
	
	z_alpha ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);
	z_theta ~ normal(0, 1);
	z_delta1 ~ normal(0, 1);
	z_delta2 ~ normal(0, 1);
	z_zeta1 ~ normal(0, 1);
	z_zeta2 ~ normal(0, 1);

  
	  

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
}
}


