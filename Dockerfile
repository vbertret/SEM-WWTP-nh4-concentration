FROM julia:1.9.3

# Set environment for cache
ENV JULIA_CPU_TARGET=generic

# Install julia package
RUN mkdir /root/SEM-WWTP-nh4-concentration
COPY . /root/SEM-WWTP-nh4-concentration/
WORKDIR /root/SEM-WWTP-nh4-concentration
RUN julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()'