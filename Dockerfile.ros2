ARG AWML_BASE_IMAGE="ghcr.io/tier4/autoware-ml-base:latest"
FROM ${AWML_BASE_IMAGE}

# Following commands are based on : https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Add the ROS 2 apt repository
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe && apt-get update && apt-get install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-argcomplete

# Create system link to fix libstdc++ import errors inside conda environment
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Set up environment variables
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
