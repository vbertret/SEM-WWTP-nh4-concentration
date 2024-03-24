FROM julia:1.9.3

# Uncomment if needed
# ENV JULIA_CPU_TARGET=generic
# ENV JULIA_DEBUG=loading

# Install the package and set it as default julia environment.
RUN mkdir /root/SEM-WWTP-nh4-concentration
COPY . /root/SEM-WWTP-nh4-concentration/
WORKDIR /root/SEM-WWTP-nh4-concentration
RUN julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()'
ENV JULIA_PROJECT=/root/SEM-WWTP-nh4-concentration/