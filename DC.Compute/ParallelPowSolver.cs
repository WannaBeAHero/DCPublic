using System.Buffers.Binary;
using System.Buffers.Text;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;

namespace DC.Compute;

public sealed class ParallelPowSolver : IPowSolver
{
    private const int AttemptsPerUpdate = 8192;

    public Task<PowResult> SolveAsync(string seed, int difficultyBits, int workerCount,
        IProgress<PowProgress>? progress, CancellationToken cancellationToken, PowTelemetry? telemetry = null)
    {
        if (seed is null || seed.Length != 6 || seed.Any(character => character is < '0' or > '9'))
            throw new ArgumentException("The seed must contain exactly six decimal digits (0–9).", nameof(seed));
        if (difficultyBits is < 1 or > PowSolver.MaximumDifficultyBits)
            throw new ArgumentOutOfRangeException(nameof(difficultyBits));
        if (workerCount is < 1 or > PowSolver.MaximumWorkers)
            throw new ArgumentOutOfRangeException(nameof(workerCount));

        return Task.Run(async () =>
        {
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var solutions = new PowSolution[PowSolver.TaskCount];
            int nextTaskId = -1;
            int completed = 0;
            long totalAttempts = 0;
            uint mask = uint.MaxValue << (32 - difficultyBits);
            var elapsed = Stopwatch.StartNew();
            telemetry?.Start();
            try
            {
                var workers = new Task[workerCount];
                for (int workerId = 0; workerId < workers.Length; workerId++)
                {
                    int laneId = workerId;
                    workers[workerId] = Task.Run(() => RunWorker(laneId));
                }

                await Task.WhenAll(workers).ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                elapsed.Stop();
                return new PowResult(Array.AsReadOnly(solutions), elapsed.Elapsed, completed, workerCount, totalAttempts);
            }
            finally
            {
                telemetry?.Finish();
            }

            void RunWorker(int workerId)
            {
                CancellationToken token = stop.Token;
                try
                {
                    while (true)
                    {
                        token.ThrowIfCancellationRequested();
                        int taskId = Interlocked.Increment(ref nextTaskId);
                        if (taskId >= solutions.Length) return;

                        int lane = telemetry?.BeginWork(taskId, workerId) ?? -1;
                        try
                        {
                            progress?.Report(new PowProgress(taskId, null,
                                Volatile.Read(ref completed), Interlocked.Read(ref totalAttempts)));
                            byte[] prefix = Encoding.UTF8.GetBytes(FormattableString.Invariant($"BSU-POW-1|{seed}|{taskId}|"));
                            var (nonce, attempts) = FindNonce(prefix, mask, token, telemetry, lane, taskId);
                            token.ThrowIfCancellationRequested();

                            var solution = new PowSolution(taskId, nonce);
                            solutions[taskId] = solution;
                            telemetry?.Solved(taskId);
                            Interlocked.Add(ref totalAttempts, attempts);
                            int done = Interlocked.Increment(ref completed);
                            progress?.Report(new PowProgress(taskId, solution, done, Interlocked.Read(ref totalAttempts)));
                        }
                        finally
                        {
                            telemetry?.EndWork(lane);
                        }
                    }
                }
                catch
                {
                    stop.Cancel();
                    throw;
                }
            }
        });
    }

    private static (ulong Nonce, long Attempts) FindNonce(ReadOnlySpan<byte> prefix, uint mask,
        CancellationToken token, PowTelemetry? telemetry, int lane, int taskId)
    {
        bool vectorized = Avx2.IsSupported;
        int batchSize = vectorized ? Sha256Batch.Width : 1;
        Span<byte> message = stackalloc byte[prefix.Length + 20];
        Span<byte> hash = stackalloc byte[32];
        prefix.CopyTo(message);
        Span<byte> blocks = stackalloc byte[Sha256Batch.Width * 64];
        Span<Vector256<uint>> schedule = stackalloc Vector256<uint>[64];
        var batch = new Sha256Batch(prefix, blocks, schedule);
        ulong nonce = 0;
        long attempts = 0;
        int pending = 0;
        try
        {
            while (true)
            {
                token.ThrowIfCancellationRequested();
                // A batch always fits, including the last one ending at ulong.MaxValue.
                for (int offset = 0; offset < AttemptsPerUpdate; offset += batchSize)
                {
                    int match;
                    if (vectorized)
                    {
                        match = batch.FindMatch(nonce, mask);
                    }
                    else
                    {
                        Utf8Formatter.TryFormat(nonce, message[prefix.Length..], out int digits);
                        SHA256.HashData(message[..(prefix.Length + digits)], hash);
                        match = (BinaryPrimitives.ReadUInt32BigEndian(hash) & mask) == 0 ? 0 : -1;
                    }

                    pending += batchSize;
                    if (match >= 0) return (nonce + (ulong)match, checked(attempts + pending));
                    if (nonce == ulong.MaxValue - (ulong)(batchSize - 1))
                        throw new InvalidOperationException($"Task {taskId} exhausted all unsigned 64-bit nonces.");
                    nonce += (ulong)batchSize;
                }

                attempts = checked(attempts + pending);
                telemetry?.AddAttempts(lane, taskId, pending);
                pending = 0;
            }
        }
        finally
        {
            telemetry?.AddAttempts(lane, taskId, pending);
        }
    }
}
