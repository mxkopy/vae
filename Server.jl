using Sockets, ClusterManagers, Distributed;

function socket(pid; host="127.0.0.1", port=2113)

    manager = LocalAffinityManager(;np=CPU_CORES, mode::AffinityMode=BALANCED, affinities=[]);

    config  = WorkerConfig()

    config.io=Base.PipeEndpoint; config.host=host; config.port=port;

    return Sockets.connect(manager, pid, config)

end





