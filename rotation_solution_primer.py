# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: qamt_env
#     language: python
#     name: qamt_env
# ---

# # QAMT Rotation Primer
#
# ## Overall Concepts
#
# ### Sensor Fusion
#
# An overview of sensor fusion would be fairly longwinded and not any more useful than the resource I used to learn about how the systems work.
# As such, referring to [Using Inertial Sensors for Position and Orientation Estimation](./Kalman_detailed.pdf), Chapter 4 in particular would be the best introduction to the equations and concepts involved.
#
# The quick version is that accelerometer, gyroscope and magnetometer (ideally) readings are combined to provide the orientation of a device.
# There are known equations indicating how any particular measurement of one of these devices is a result of the true physics involved plus some noise (Chapter 3.3 to Chapter 3.7 in Using Inertial Sensors).
# These equations are used in a few different methods of determining what the orientation and pose of the craft for which the measurements are taken, most of which make use of the Kalman Filter methodology.
# Broadly defined, Kalman filter solutions take a current set of measurements, the estimated noise on those measurements and the theoretical model and are used to make a guess at what the next incoming state would be.
# This guess is compared to the next set of measurements, and the difference between the two is used to tune the alogirthm such that subsequent guesses and measurements have a decreasing difference.
#
# A great resource I used to help gain the basic understanding of Kalman filter operation is at [this site](https://www.kalmanfilter.net/default.aspx).
# The complexity of our problem quickly outstrips this source however.
# The basic Kalman Filter only works with linear equations, and non-linearities cannot be properly processed.
# This leads to either Extended Kalman filters (EKF), which use Taylor expansion of the Kalman filter equations to allow a linear estimation of non-linear functions, or to Unscented Kalman filters (UKF), which do not linearize equations but rather take a series of "sigma points" from the current measurement, transform them based on the non-linear funcitons, then find a distribution which best fits the transformed points and update state based on these points.
# The EKF is best used for moderately non-linear problems, and the UKF for highly non-linear problems.
#
# There have been a plethora of papers published on this topic.
# The ones I have been referring to most heavily for our implementations are [Pi-Invariant Unscented Kalman Filter for Sensor Fusion](./condomines_piinvariantUKF.pdf), [9-DOF IMU_Bases Attitude and Heading Estimation Using an Extended Kalman Filter with Bias Consideration](./imubased_attitude_heading_ekf.pdf), [Quaternion-Based Robust Attitude Estimation Using an Adaptive Unscented Kalman Filter](./qraukf_paper.pdf) and [VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic Disturbance Rejection](./vqf_paper.pdf)
#
#

# ### Using the Rotation Solution to Obtain Mag Data for a Flight
#
# Under the assumption that the sensor fusion algorithm obtains valid orientations, the next steps are:
#
# - Use GPS or RTK Lib solution to obtain the International Geomagnetic Reference Field (IGRF) at each of the locations of the bird during the flight
# - Use this information to generate a correction for geomagnetic and geographic North (VQF and magnetometer based corrections will typically generate an orientation pointing towards magnetic north).
# - Use the orientation information to rotate the IGRF field into the bird reference frame
# - Use the Procrustes method to determine the offset, scaling factor and rotation of the SQUID data to align it with the bird reference frame
# - Maybe apply the motion noise compensation algorithm used by Supracon (more in next section)
# - Use the rotation solution to rotate the modified SQUID data back into the Earth reference frame (ideally will look similar to the IGRF field, with disturbances)
#
# The first few sections of [Rotation_Solution_Update](./Rotation_Solution_Update.ipynb) give a good overview on what this process looks like, but I can demonstrate as needed as well.

# ## Current Workflow
#
# The current workflow makes use of the VQF code-base for sensor fusion.
# The following flowchart shows all current steps used:
#
# ```{figure} ./rotation_solution_flowchart.png
# :name: rotation_flowchart
# :width: 700
#
# Rotation Solution flowchart. Note that the VQF block could theoretically be replaced with any sensor fusion algorithm.
# ```
#

# ## Current Issues
#
# This Jupyter notebook summarizes many of the existing issues that we were experiencing: [Rotation Solution Update](./Rotation_Solution_Update.ipynb)
#
# Many of these have been solved, though there are still issues with the fact that the motion noise compensation (mnc) algorithm used by Supracon is not giving us the same results as it was for Markus, even when we use Markus' rotation solution.
#
# The other issue is that I cannot get the VQF solution and that obtained by Markus to match.
# This remains true even if I use Markus' solution as the minimization target for the tuning parameters to VQF.
# This is our major roadblock now.

# ## Approaches Tried
#
# ### Legacy Code (Python and Cython)
#
# First and foremost, I am not certain what happened with the code Hugo left behind when I started.
# I could not get the code to work as it was, and I suspect that he was in the middle of some breaking changes to the overall processing chain when he left.
# In light of this, I ported over as much of the existing kalman filter code over as it is, while removing the steps that sought to write results to very specific folder paths and hdf5 files as interim results, instead keeping it all in local memory.
# The code as it exists right now takes about 5 days to run the Namibia data file, which comprises about 3 hours of flight data.
#
# The main reason for this is that Kalman filter methods necessarily use for-loops to run through all the data points, which is incredibly slow in Python.
# This is also the reason that my attempts to write a Cython equivalent (conversion to C to use in Python) led to very little optimization (4 days instead of 5); most of the functions and math used in the Kalman filter interact with Python libraries and Python objects, and thus are not translated into C.
#
# It is worth noting that the existing code developed by Hugo is a recreation of the QRAUKF method in Python, and the original is freely available as a MATLAB repository, and could potentially be used to debug/reference/check.
#
# The results from this existing code are not coherent and resemble noise more than any sort of actual solution.
# I can see from reports Hugo made that the solution did work at some point, even if it was slow, but I cannot determine where the solution has broken.
#
# A large part of this may be due to the need to specify the noise parameters for each measurement type and the expected variance in the solution.
# If the noise parameters are not set correctly, the Kalman filter will happily give nonsensical solutions that are technically within the noise bounds.
# These parameters would have to be tuned using a simplified, known orientation/location test most likely.
#
# ### VQF
#
# This is the currently implemented sensor fusion algorithm used in our processing code.
# Despite the recent discovery that the solution can be tuned by changing a few parameters and obtaining results very close to what Supracon obtains, there is still some residual drift remaining in our solution.
# {ref}`vqf_sol` shows the current state of our best-case solution.
#
# This means that when I use the VQF solution, there is an artificial slope remaining in many of our components that should not be there.
# In reading through the documentation for VQF, it appears that the algorithm may not do well in situations where there is high magnetic disturbance.
# If so, this makes it fundamentally ill-posed for solving our solution due to the nature of QAMT measurements.
#
# ```{figure} ./Figures_for_documents/primer/vqf_solution.png
# :name: vqf_sol
#
# VQF Solution at present time
# ```
#

# ### A Posteriori Method
#
# One of the methods of solving this issue that is discussed in Chapter 4 of Using Inertial Sensors is to take the whole series of measurements and their expected distributions to find the most probable state.
# This type of methodology was very similar to the work I had done in my MSc and PhD work, so I thought I would try this as well.
# The ```pose_estimate_lib.py``` script is the result of this investigation, where I learned that the method requires a sparse matrix of VERY LARGE PROPORTIONS to be inverted, which requires more memory than is feasible.
# Breaking the flight into smaller, more manageable chunks of the full flight is also not a good idea, as each line or segment would be optimized separately and is almost certain to give a series of results that are discontinuous.
# I have since abandoned this course of action, realizing this is why almost every paper on the topic uses some kind of Kalman filter rather than an inverse solution.

# ### MATLAB Sensor Fusion and Tracking Toolbox
#
# Glenn has obtained a license for us to use the Sensor Fusion and Tracking Toolbox for MATLAB.
# This looks to be an active, well-established and reputable set of tools designed specifically for the type of problem we face.
#
# Like the Python/Cython solution however, this is also very slow and gives results that are non-physical, likely due to incorrect noise parameters.
# See {ref}`matlab_sol` for an example of the kind of current result we are seeing.
#
# This method as well may require us to do a series of non-flight calibration data sets where we know the true position and orientation of the bird so we can do proper tuning.
# The tuning of filters used by the Tracking Toolbox is done with built-in functions.
#
# This is an example of the most recent run of the insfilterMARG filter in the Tracking Toolbox:
# ```{figure} ./Figures_for_documents/primer/matlab_solution.png
# :name: matlab_sol
#
# MATLAB Solution using the ```insfilterMARG``` function in the Sensor Fusion and Tracking Toobox.
# ```
# As we can see, it's essentially noise at this time.

# ## Moving Forward
#
# The MATLAB library was the latest attempt made at solving this problem, and various tuning parameter checks are being done with VQF (with low confidence).
# I beleive that our next best step may be to switch to C as a method of obtaining quicker turn around.
# The Tracking Toolbox was designed to work with MATLAB Coder, which allows the direct translation of the MATLAB algorithms into C code, but we do not have a license for it.
# I don't know if that's a path we want to go down or if we should start from scratch.
# I'm open to ideas, but do not have a lot of experience with C code.


