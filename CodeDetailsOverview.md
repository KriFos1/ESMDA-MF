Where the prior is built between logging points
• sequential_matern_1d.py: carry_posterior() (line 93) turns the previous posterior into the next prior.
    1. reindex_window() shifts the window by whole cells and copies the edge value into cells with no source.
    2. relax_to_prior() is an Ornstein-Uhlenbeck step toward the Matérn prior. It runs only when --memory-ft gives rho < 1.
       --memory-mean-factor (default 4) lets the ensemble mean (local background) outlive the spread (relax_to_prior(rho_mean=...)): the mean relaxes with memory * factor, the spread with memory. Recommended: --memory-ft 50 with the default factor.
       The ahead factor, taper, latent factor and surprise options were removed after the fault studies showed no benefit.
    3. prior.condition_entered() (matern_prior.py:155) fills the newly entered cells from the Matérn prior, conditioned on the cells that were kept.
• main() calls it around lines 470–484. The first point is a fresh draw from the prior (prior.sample).
• 1D → 2D hand-off:
    ◦ chain_laplace.py: export_problem() and fitted_prior() define the 2D prior. It is built from the carried ensemble through the --two-d-prior choice (matern_cellwise or ensemble_precision).
    ◦ ridge_propagator() and lateral_update() (lines 211–246 of sequential_matern_1d.py) control how a 1D update spreads laterally.
    ◦ laplace_posterior() builds the ensemble that returns from 2D to 1D. 

Subset around transitions: Use --start and --stop. The first point of a run starts from a fresh prior draw.




Duplication with the cluster
• The controller and the 1D chain run only on your machine. Only the 2D MAP members run on the cluster.
• bundle packs the project modules and Dashboard/src/pxwf into a tarball. run_chain_laplace_loop.py uploads it to --hpc-root for each point. So you don't maintain a second copy of the code on the cluster. A changed prior generator is picked up automatically.
• The prior code stays local. The cluster only needs the prior ensemble that arrives inside the tarball.
• A folder next to krfo, such as /cluster/home/mlie/, is fine. All paths are flags, so nothing touches krfo.
• Exception: the bundle's module list (MODULES in pilot_rml_index45.py) must include any new module that the cluster's MAP stage imports.

Here is the local run from start to finish. 
Setup, once per terminal (the PyCharm terminal with the venv does this for you)
cd ~/3DgigNew/ESMDA-MF
source ../venv-pet-latest/bin/activate      # sets thread counts and $BM; pxwf is on the path via a .pth file
1. Observations (output is in Ex1/)
    python -B write_data.py --benchmark $BM --output-dir Ex1
    The output directory must be new each time.
2. Reference manifest (runs/benchmark3-reference-bedding/run.json) It points to Ex1/data.pkl and Ex1/var.pkl. If you write new data elsewhere, edit the data and variance paths in this file.
3. 1D chain until the first gate (use a new NAME for each run)
NAME=my-run-1
   python -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
     --output-dir runs/$NAME --stop 281 --ensemble-size 250 --seed 2031 --mda-steps 10 \
     --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
     --memory-ft 50 --gate-threshold 25 --on-gate stop --analysis enif \
     --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
   It runs about 25 s per point. For a subset of logging points, add --start N --stop M (--stop is exclusive). In my 281-point test, the gate first opened at index 30. A lower --gate-threshold, such as 1.5, makes the gate open sooner.
4. Controller: 1D chain plus 2D at the gates
python -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop 281 \
  --source references/map-source --label $(date +%Y%m%d) --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir $HOME/3DgigNew/uta2d_v2/adjoint-2d-v2-orient --orientation bedding \
  --executor node --host localhost --hpc-root $HOME/pxwf-runs-local --hpc-python $(which python) \
  --node-threads 3 --node-concurrency 4 --node-hdfql $HOME/3DgigNew/uta2d_v2/adjoint-2d-v2-orient \
  --oneapi $HOME/3DgigNew/uta2d_v2/oneapi-local-vars.sh
5. Use the same --stop (and --start, if you used one) as in step 3. Each 2D point took 6–9 minutes in my test.

6. Useful extras
• Pause: touch runs/$NAME/PAUSE stops the controller at the next point boundary. Remove the file and rerun the same command to continue.
• Curtain plot: python -B plot_curtain.py --run-dir runs/$NAME (for --follow, add --interval 180).
• Output: results are in runs/$NAME/inversion_results_assim<i>.pkl. The 2D runs are in runs/chain-laplace-idx<i>-<label>.

7. Before long runs, check that ssh localhost hostname still works without a password.

To watch the progress (second terminal)
tail -f ~/pxwf-runs-local/chain-laplace-idx030-20261029/run/member-000.log
That log is empty until the first iteration finishes, so don't worry if it stays empty for a few minutes. To stop the run safely, use touch runs/my-run-4/PAUSE instead of Ctrl+C.


%%%%%%%%%%%%%%%%%%%% TERMINAL RUN on HPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd ~/3DgigNew/ESMDA-MF && source ../venv-pet-latest/bin/activate
NAME=hpc-test
python -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --stop 281 --ensemble-size 250 --seed 2031 --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 --gate-threshold 25 --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25 \
  --memory-ft 50
python -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop 1 \
  --source references/map-source --label hpc1 --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=1 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python

Stop hpc run:
1. Press Ctrl+C in the terminal running run_chain_laplace_loop.py. This stops polling, but does not cancel jobs already submitted to Slurm.
2. Cancel the cluster jobs:
ssh HPC squeue -u mlie                  # list your jobs and IDs
ssh HPC scancel <jobid>                 # cancel one (an array ID cancels all its tasks)
ssh HPC scancel -u mlie                 # cancel everything you own
ssh HPC "scancel -n rml-array"          # only by job name, if you have other jobs


%%%%%%%%%%%%%%%%%%%% TERMINAL RUN on LOCAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd ~/3DgigNew/ESMDA-MF && source ../venv-pet-latest/bin/activate
NAME=my-run-4
python -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --stop 281 --ensemble-size 250 --seed 2031 --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 --gate-threshold 25 --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
python -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop 281 \
  --source references/map-source --label $(date +%Y%m%d) --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir $HOME/3DgigNew/uta2d_v2/adjoint-2d-v2-orient --orientation bedding \
  --executor node --host localhost --   hpc-root $HOME/pxwf-runs-local --hpc-python $(which python) \
  --node-threads 3 --node-concurrency 4 --node-hdfql $HOME/3DgigNew/uta2d_v2/adjoint-2d-v2-orient \
  --oneapi $HOME/3DgigNew/uta2d_v2/oneapi-local-vars.sh     

%%%%%%%%%%%%%%%%%%%% PLOTTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plotting and a GIF both work. The curtain script plot_curtain.py already existed in ESMDA-MF, and I added a --gif option to it. I rendered runs/my-run-4 and the GIF wrote without errors. I did not open the GIF to check it by eye. I did view the still image.
Static image (runs/my-run-4/curtain.png):
cd ~/3DgigNew/ESMDA-MF && source ../venv-pet-latest/bin/activate
python plot_curtain.py --run-dir runs/my-run-4 --paint nearest-bit --colormap hue --uncertainty-lightness
It shows:
• the reference log Rh;
• the estimated Rh and Rv;
• the well path, with 1D and 2D points marked;
• the per-point misfit with the gate threshold.
On your run, the resistive body past about 200 ft is recovered.
GIF (runs/my-run-4/curtain.gif, 49 frames, about 1.7 MB):
python plot_curtain.py --run-dir runs/my-run-4 --paint nearest-bit --colormap hue --uncertainty-lightness --gif runs/my-run-4/curtain.gif [--gif-fps 2] [--dpi 80]
Each frame adds one logging point. It takes about a minute at --dpi 80.
During the run: Yes, open a second terminal and run:
python plot_curtain.py --run-dir runs/<name> --follow --interval 120 --paint nearest-bit --colormap hue
It re-renders curtain.png whenever a new checkpoint appears. For a GIF, run --gif once the chain has finished, or at any time to cover the points done so far.
Benchmark path: plot_curtain.py defaults to ../Dashboard/benchmarks/Benchmark-3. That path exists here as a symlink, so you can drop --benchmark. I passed it explicitly.
Cosmetic issue: The colorbar tick labels overlap the neighbouring axis labels in the top row. I haven't changed this.