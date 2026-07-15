#!/bin/bash
# Driver for selective C/R validation. Runs inside the GPU pod.
# Expects: /work/gpu-cr (fixed source tree), /work/test (test programs).
set -x

WORK=/work
BUILD=$WORK/gpu-cr/build
export EXPORT_FILE_PATH=/ckpt
mkdir -p $EXPORT_FILE_PATH

CR=$BUILD/cr_client
SO=$BUILD/vGPU-NVIDIA.so

vram_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

wait_file() {
    local f=$1; local timeout=${2:-120}
    for i in $(seq 1 $((timeout * 5))); do
        [ -f "$f" ] && return 0
        sleep 0.2
    done
    echo "TIMEOUT waiting for $f"
    return 1
}

run_ctest() {
    echo "=================== TEST 1: C++ selective C/R ==================="
    export TEST_DIR=/tmp/seltest
    rm -rf $TEST_DIR && mkdir -p $TEST_DIR
    rm -f $EXPORT_FILE_PATH/control* $EXPORT_FILE_PATH/ckpt-*

    LD_PRELOAD=$SO $WORK/test/selective_test >/tmp/seltest-app.log 2>&1 &
    APP_PID=$!

    wait_file $TEST_DIR/ready || return 1
    read PID PTRA PTRB PTRC SIZE < $TEST_DIR/ready
    echo "app pid=$PID A=$PTRA B=$PTRB C=$PTRC size=$SIZE"

    $CR -i -p $PID || return 1

    USED_BEFORE=$(vram_used)
    $CR -c -s ${PTRA}:${SIZE} -p $PID || return 1
    sleep 1
    USED_AFTER_CKPT=$(vram_used)
    echo "VRAM used: before=$USED_BEFORE MB, after ckpt=$USED_AFTER_CKPT MB"

    touch $TEST_DIR/ckpt_done
    wait_file $TEST_DIR/midcheck || return 1
    echo "midcheck: $(cat $TEST_DIR/midcheck)"

    $CR -r -s ${PTRA}:${SIZE} -p $PID || return 1
    touch $TEST_DIR/restore_done

    wait $APP_PID
    APP_RC=$?
    echo "---- app log tail ----"; tail -30 /tmp/seltest-app.log
    echo "RESULT(C++): $(cat $TEST_DIR/result 2>/dev/null) rc=$APP_RC"
    [ "$APP_RC" = "0" ]
}

run_torchtest() {
    echo "=================== TEST 2: PyTorch selective C/R ==================="
    export TEST_DIR=/tmp/seltest-torch
    rm -rf $TEST_DIR && mkdir -p $TEST_DIR
    rm -f $EXPORT_FILE_PATH/control* $EXPORT_FILE_PATH/ckpt-*

    PYTORCH_NO_CUDA_MEMORY_CACHING=1 LD_PRELOAD=$SO \
        python3 $WORK/test/torch_selective_test.py >/tmp/seltest-torch-app.log 2>&1 &
    APP_PID=$!

    wait_file $TEST_DIR/ready 300 || { tail -50 /tmp/seltest-torch-app.log; return 1; }
    read PID PTR SIZE < $TEST_DIR/ready
    echo "torch pid=$PID ptr=$PTR size=$SIZE"

    $CR -i -p $PID || return 1

    USED_BEFORE=$(vram_used)
    $CR -c -s ${PTR}:${SIZE} -p $PID || return 1
    sleep 1
    USED_AFTER_CKPT=$(vram_used)
    echo "VRAM used: before=$USED_BEFORE MB, after ckpt=$USED_AFTER_CKPT MB"

    touch $TEST_DIR/ckpt_done
    wait_file $TEST_DIR/midcheck 60 || { tail -50 /tmp/seltest-torch-app.log; return 1; }
    echo "midcheck: $(cat $TEST_DIR/midcheck)"

    $CR -r -s ${PTR}:${SIZE} -p $PID || return 1
    touch $TEST_DIR/restore_done

    wait $APP_PID
    APP_RC=$?
    echo "---- torch app log tail ----"; tail -40 /tmp/seltest-torch-app.log
    echo "RESULT(torch): $(cat $TEST_DIR/result 2>/dev/null) rc=$APP_RC"
    [ "$APP_RC" = "0" ]
}

RC1=1; RC2=1
run_ctest && RC1=0
run_torchtest && RC2=0

echo "================================================="
echo "FINAL: C++ test $([ $RC1 = 0 ] && echo PASS || echo FAIL), torch test $([ $RC2 = 0 ] && echo PASS || echo FAIL)"
exit $((RC1 + RC2))
