FROM slimerl/slime@sha256:c95eb968649ee8e2a567c500df5a5643c9f85951622646fe6e08244d939e4b94

# Install the timeslice client library
RUN pip install --no-cache-dir "grpcio>=1.81.1" "protobuf>=7.35"
RUN pip install --no-cache-dir "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git@11a661f#subdirectory=pkg/client/python"

# Replace entire Slime installation with our fork (phase callback support)
RUN pip install --no-cache-dir gitpython
RUN git clone --depth 1 -b feat/phase-callbacks https://github.com/aishukamal/slime.git /tmp/slime-fork && \
    rm -rf /root/slime && \
    mv /tmp/slime-fork /root/slime && \
    cd /root/slime && pip install --no-cache-dir -e .

# Reinstall grpcio/protobuf after Slime (Slime deps may downgrade them)
RUN pip install --no-cache-dir "grpcio>=1.81.1" "protobuf>=7.35"

# Install the timeslice-slime callback package
COPY timeslice-slime/ /tmp/timeslice-slime/
RUN pip install --no-cache-dir /tmp/timeslice-slime/ && rm -rf /tmp/timeslice-slime/
