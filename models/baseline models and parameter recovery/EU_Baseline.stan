data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of participants
	int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=-1,upper=1> cho[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt
  real oa[N,4];
  real ob[N,4];
  real pa[N,4];
  real pb[N,4];
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
parameters {
	real mu_alpha;
	real mu_theta;
	real mu_threshold;
	real mu_ndt;

	real<lower=0> sd_alpha;
	real<lower=0> sd_theta;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;
	
	
	real z_theta[L];
	real z_threshold[L];
	real z_alpha[L];
	real z_ndt[L];
}
transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real ua[N];	
	real ub[N];	
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt

	real<lower=0> alpha_sbj[L];
	real<lower=0> theta_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];

	real transf_mu_alpha;
	real transf_mu_theta;
	real transf_mu_threshold;
	real transf_mu_ndt;

	transf_mu_alpha = log(1+ exp(mu_alpha));						// for the output
	transf_mu_theta = log(1+ exp(mu_theta));					
	transf_mu_threshold = log(1+ exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
	  alpha_sbj[l] = log(1 + exp(mu_alpha + z_alpha[l]*sd_alpha));
		theta_sbj[l] = log(1 + exp(mu_theta + z_theta[l]*sd_theta));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		ua[n] = pa[n,1] * pow(oa[n,1],alpha_sbj[participant[n]]) + pa[n,2] * pow(oa[n,2],alpha_sbj[participant[n]]) + pa[n,3] * pow(oa[n,3],alpha_sbj[participant[n]]) + pa[n,4] * pow(oa[n,4],alpha_sbj[participant[n]]);
		ub[n] = pb[n,1] * pow(ob[n,1],alpha_sbj[participant[n]]) + pb[n,2] * pow(ob[n,2],alpha_sbj[participant[n]]) + pb[n,3] * pow(ob[n,3],alpha_sbj[participant[n]]) + pb[n,4] * pow(ob[n,4],alpha_sbj[participant[n]]);
		drift_t[n] = theta_sbj[participant[n]] * (ub[n] - ua[n]);
		drift_ll[n] = drift_t[n]*cho[n];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];
	}
}
model {
	mu_alpha ~ normal(1, 5);
	mu_theta ~ normal(1, 5);
	mu_threshold ~ normal(1, 3);
	mu_ndt ~ normal(1, 1);
	


	sd_alpha ~ normal(0, 5);
	sd_theta ~ normal(0, 5);
	sd_threshold ~ normal(0,3);
	sd_ndt ~ normal(0,1);


	
	z_alpha ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);
	z_theta ~ normal(0, 1);

	

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
}
}
