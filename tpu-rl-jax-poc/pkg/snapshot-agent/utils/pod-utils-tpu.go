package utils

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

const (
	JobIDLabel = "timeslice.io/job-id"
)

var (
	GetK8sClient = func() (kubernetes.Interface, error) {
		config, err := rest.InClusterConfig()
		if err != nil {
			return nil, err
		}
		return kubernetes.NewForConfig(config)
	}

	IsPIDInPodCgroupFunc = isPIDInPodCgroup
)

func GetLocalPods(ctx context.Context, jobID string) ([]corev1.Pod, error) {
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		return nil, fmt.Errorf("NODE_NAME environment variable not set")
	}

	clientset, err := GetK8sClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
		LabelSelector: fmt.Sprintf("%s=%s", JobIDLabel, jobID),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods on node %s: %w", nodeName, err)
	}

	return podList.Items, nil
}

// GetPodPIDs returns the host-namespace PIDs of all TPU-context-holding processes belonging to the specified pod.
// Discovers PIDs by scanning for /run/tpu_hal_<PID>.sock files and cross-referencing with pod cgroups.
func GetPodPIDs(ctx context.Context, podName, namespace string) ([]int, error) {
	logger := klog.FromContext(ctx)

	podUID, err := getPodUID(ctx, podName, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod UID: %w", err)
	}

	// Find all TPU HAL sockets
	sockets, err := filepath.Glob("/run/tpu_hal_*.sock")
	if err != nil {
		return nil, fmt.Errorf("failed to glob TPU HAL sockets: %w", err)
	}

	var pids []int
	for _, sock := range sockets {
		base := filepath.Base(sock)
		// Extract PID from tpu_hal_<PID>.sock
		pidStr := strings.TrimPrefix(base, "tpu_hal_")
		pidStr = strings.TrimSuffix(pidStr, ".sock")
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			continue
		}

		inCgroup, err := IsPIDInPodCgroupFunc(pid, podUID)
		if err != nil {
			logger.V(2).Info("Failed to check cgroup for PID", "pid", pid, "error", err)
			continue
		}

		if inCgroup {
			pids = append(pids, pid)
			logger.Info("Found TPU process in pod", "pid", pid, "pod", podName, "socket", sock)
		}
	}

	return pids, nil
}

func getPodUID(ctx context.Context, podName, namespace string) (string, error) {
	clientset, err := GetK8sClient()
	if err != nil {
		return "", err
	}
	pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return string(pod.UID), nil
}

func isPIDInPodCgroup(pid int, podUID string) (bool, error) {
	return IsPIDInPodCgroupInternal(fmt.Sprintf("/proc/%d/cgroup", pid), podUID)
}

func IsPIDInPodCgroupInternal(cgroupPath, podUID string) (bool, error) {
	f, err := os.Open(cgroupPath)
	if err != nil {
		return false, err
	}
	defer f.Close()

	podUIDUnderscores := strings.ReplaceAll(podUID, "-", "_")

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, podUID) || strings.Contains(line, podUIDUnderscores) {
			return true, nil
		}
	}
	return false, scanner.Err()
}
