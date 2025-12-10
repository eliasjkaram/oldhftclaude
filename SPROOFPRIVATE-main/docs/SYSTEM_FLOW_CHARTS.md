# Alpaca-MCP System Flow Charts

## Table of Contents
1. [Main Trading Flow](#main-trading-flow)
2. [Order Execution Flow](#order-execution-flow)
3. [ML Training Pipeline](#ml-training-pipeline)
4. [Risk Management Flow](#risk-management-flow)
5. [Options Strategy Flow](#options-strategy-flow)
6. [Data Processing Flow](#data-processing-flow)
7. [AI Algorithm Decision Flow](#ai-algorithm-decision-flow)
8. [Market Event Response Flow](#market-event-response-flow)

## Main Trading Flow

```mermaid
flowchart TD
    Start([System Start]) --> Init[Initialize Components]
    Init --> LoadConfig[Load Configuration]
    LoadConfig --> ConnectData[Connect Data Sources]
    
    ConnectData --> MainLoop{Trading Hours?}
    MainLoop -->|Yes| FetchData[Fetch Market Data]
    MainLoop -->|No| Sleep[Sleep Until Market Open]
    Sleep --> MainLoop
    
    FetchData --> UpdatePortfolio[Update Portfolio State]
    UpdatePortfolio --> RunStrategies[Run Trading Strategies]
    
    RunStrategies --> ParallelExec{Execute in Parallel}
    ParallelExec --> MLPred[ML Predictions]
    ParallelExec --> QuantumAnalysis[Quantum Analysis]
    ParallelExec --> SwarmIntel[Swarm Intelligence]
    ParallelExec --> OptionsArb[Options Arbitrage]
    ParallelExec --> HFTScanning[HFT Scanning]
    
    MLPred --> SignalAgg[Signal Aggregation]
    QuantumAnalysis --> SignalAgg
    SwarmIntel --> SignalAgg
    OptionsArb --> SignalAgg
    HFTScanning --> SignalAgg
    
    SignalAgg --> RiskCheck{Risk Check Pass?}
    RiskCheck -->|No| LogReject[Log Rejection]
    LogReject --> MainLoop
    
    RiskCheck -->|Yes| OrderGen[Generate Orders]
    OrderGen --> OrderValid{Validate Orders?}
    OrderValid -->|No| LogError[Log Validation Error]
    LogError --> MainLoop
    
    OrderValid -->|Yes| ExecuteOrders[Execute Orders]
    ExecuteOrders --> UpdatePositions[Update Positions]
    UpdatePositions --> UpdateMetrics[Update Performance Metrics]
    UpdateMetrics --> CheckEOD{End of Day?}
    
    CheckEOD -->|No| MainLoop
    CheckEOD -->|Yes| DailyReport[Generate Daily Report]
    DailyReport --> Shutdown([System Shutdown])
```

## Order Execution Flow

```mermaid
flowchart TD
    Signal([Trading Signal]) --> CreateOrder[Create Order Object]
    CreateOrder --> OrderType{Order Type?}
    
    OrderType -->|Market| MarketOrder[Market Order Logic]
    OrderType -->|Limit| LimitOrder[Limit Order Logic]
    OrderType -->|Options Spread| SpreadOrder[Spread Order Logic]
    OrderType -->|Smart| SmartOrder[Smart Order Router]
    
    MarketOrder --> PreTradeCheck[Pre-Trade Risk Check]
    LimitOrder --> PreTradeCheck
    SpreadOrder --> PreTradeCheck
    SmartOrder --> PreTradeCheck
    
    PreTradeCheck --> RiskPass{Risk Check Pass?}
    RiskPass -->|No| RejectOrder[Reject Order]
    RejectOrder --> NotifyRisk[Notify Risk Manager]
    NotifyRisk --> End([End])
    
    RiskPass -->|Yes| CheckCapital{Sufficient Capital?}
    CheckCapital -->|No| RejectCapital[Reject - Insufficient Capital]
    RejectCapital --> End
    
    CheckCapital -->|Yes| RouteOrder[Route to Exchange]
    RouteOrder --> AlpacaAPI{Trading Mode?}
    
    AlpacaAPI -->|Live| LiveExecution[Live Execution]
    AlpacaAPI -->|Paper| PaperExecution[Paper Execution]
    AlpacaAPI -->|Custom Paper| CustomPaper[Custom Paper Trading]
    
    LiveExecution --> OrderStatus{Order Status}
    PaperExecution --> OrderStatus
    CustomPaper --> SimulateExecution[Simulate Execution]
    SimulateExecution --> OrderStatus
    
    OrderStatus -->|Filled| RecordFill[Record Fill]
    OrderStatus -->|Partial| RecordPartial[Record Partial Fill]
    OrderStatus -->|Rejected| RecordReject[Record Rejection]
    OrderStatus -->|Cancelled| RecordCancel[Record Cancellation]
    
    RecordFill --> UpdatePosition[Update Position]
    RecordPartial --> UpdatePosition
    RecordReject --> NotifyStrategy[Notify Strategy]
    RecordCancel --> NotifyStrategy
    
    UpdatePosition --> UpdatePnL[Update P&L]
    UpdatePnL --> UpdateRisk[Update Risk Metrics]
    UpdateRisk --> BroadcastUpdate[Broadcast Position Update]
    BroadcastUpdate --> End
    NotifyStrategy --> End
```

## ML Training Pipeline

```mermaid
flowchart TD
    Start([ML Training Start]) --> LoadHistorical[Load Historical Data from MinIO]
    LoadHistorical --> DataQuality{Data Quality Check}
    
    DataQuality -->|Fail| DataCleaning[Data Cleaning]
    DataCleaning --> DataQuality
    DataQuality -->|Pass| FeatureEng[Feature Engineering]
    
    FeatureEng --> Features{134 Features}
    Features --> TechIndicators[Technical Indicators]
    Features --> MarketMicro[Market Microstructure]
    Features --> Sentiment[Sentiment Features]
    Features --> MacroEcon[Macro Economic]
    Features --> OptionGreeks[Option Greeks]
    
    TechIndicators --> FeatureMatrix[Create Feature Matrix]
    MarketMicro --> FeatureMatrix
    Sentiment --> FeatureMatrix
    MacroEcon --> FeatureMatrix
    OptionGreeks --> FeatureMatrix
    
    FeatureMatrix --> MarketRegime[Market Regime Labeling]
    MarketRegime --> CycleLabels{Market Cycles}
    
    CycleLabels --> DotCom[Dot-Com Bubble]
    CycleLabels --> Financial08[2008 Crisis]
    CycleLabels --> Covid[COVID Crash]
    CycleLabels --> AIBoom[AI Boom]
    
    DotCom --> LabeledData[Labeled Dataset]
    Financial08 --> LabeledData
    Covid --> LabeledData
    AIBoom --> LabeledData
    
    LabeledData --> SplitData[Train/Validation/Test Split]
    SplitData --> ModelSelection{Model Selection}
    
    ModelSelection --> RandomForest[Random Forest]
    ModelSelection --> XGBoost[XGBoost]
    ModelSelection --> LSTM[LSTM Networks]
    ModelSelection --> Transformer[Transformer Models]
    ModelSelection --> Ensemble[Ensemble Methods]
    
    RandomForest --> Training[Model Training]
    XGBoost --> Training
    LSTM --> Training
    Transformer --> Training
    Ensemble --> Training
    
    Training --> CrossVal[Time Series Cross-Validation]
    CrossVal --> Metrics{Performance Metrics}
    
    Metrics --> Accuracy[Accuracy/F1 Score]
    Metrics --> Sharpe[Sharpe Ratio]
    Metrics --> MaxDD[Max Drawdown]
    Metrics --> ProfitFactor[Profit Factor]
    
    Accuracy --> ModelEval[Model Evaluation]
    Sharpe --> ModelEval
    MaxDD --> ModelEval
    ProfitFactor --> ModelEval
    
    ModelEval --> PassThreshold{Pass Threshold?}
    PassThreshold -->|No| HyperTuning[Hyperparameter Tuning]
    HyperTuning --> Training
    
    PassThreshold -->|Yes| EdgeCaseTest[Edge Case Testing]
    EdgeCaseTest --> EdgeScenarios{23 Edge Cases}
    
    EdgeScenarios --> FlashCrash[Flash Crash]
    EdgeScenarios --> CircuitBreaker[Circuit Breakers]
    EdgeScenarios --> Halts[Trading Halts]
    EdgeScenarios --> Gaps[Gap Opens]
    
    FlashCrash --> RobustnessCheck[Robustness Validation]
    CircuitBreaker --> RobustnessCheck
    Halts --> RobustnessCheck
    Gaps --> RobustnessCheck
    
    RobustnessCheck --> FinalValidation{Final Validation}
    FinalValidation -->|Fail| ModelRefinement[Refine Model]
    ModelRefinement --> Training
    
    FinalValidation -->|Pass| SaveModel[Save Model]
    SaveModel --> ModelRegistry[Model Registry]
    ModelRegistry --> Deploy[Deploy to Production]
    Deploy --> Monitor[Monitor Performance]
    Monitor --> End([End])
```

## Risk Management Flow

```mermaid
flowchart TD
    Start([Risk Check Start]) --> LoadPosition[Load Current Positions]
    LoadPosition --> LoadLimits[Load Risk Limits]
    LoadLimits --> NewOrder[New Order Details]
    
    NewOrder --> PreTradeChecks{Pre-Trade Checks}
    
    PreTradeChecks --> PositionLimit[Position Limit Check]
    PreTradeChecks --> VaRCheck[VaR Calculation]
    PreTradeChecks --> MarginCheck[Margin Requirements]
    PreTradeChecks --> CorrelationCheck[Correlation Analysis]
    PreTradeChecks --> StressTest[Stress Testing]
    
    PositionLimit --> LimitOK{Within Limits?}
    LimitOK -->|No| RejectPosition[Reject - Position Limit]
    LimitOK -->|Yes| VaRCalc[Calculate Portfolio VaR]
    
    VaRCheck --> VaRCalc
    VaRCalc --> VaROK{VaR Acceptable?}
    VaROK -->|No| RejectVaR[Reject - VaR Exceeded]
    VaROK -->|Yes| MarginCalc[Calculate Margin Impact]
    
    MarginCheck --> MarginCalc
    MarginCalc --> MarginOK{Margin Sufficient?}
    MarginOK -->|No| RejectMargin[Reject - Insufficient Margin]
    MarginOK -->|Yes| CorrelationCalc[Calculate Correlations]
    
    CorrelationCheck --> CorrelationCalc
    CorrelationCalc --> CorrelationOK{Correlation Risk OK?}
    CorrelationOK -->|No| RejectCorr[Reject - High Correlation]
    CorrelationOK -->|Yes| StressCalc[Run Stress Scenarios]
    
    StressTest --> StressCalc
    StressCalc --> StressScenarios{Stress Scenarios}
    
    StressScenarios --> MarketCrash[Market Crash -20%]
    StressScenarios --> VolSpike[Volatility Spike]
    StressScenarios --> LiquidityCrisis[Liquidity Crisis]
    StressScenarios --> Correlation1[Correlation = 1]
    
    MarketCrash --> ScenarioResults[Aggregate Results]
    VolSpike --> ScenarioResults
    LiquidityCrisis --> ScenarioResults
    Correlation1 --> ScenarioResults
    
    ScenarioResults --> StressOK{All Scenarios Pass?}
    StressOK -->|No| RejectStress[Reject - Stress Test Failed]
    StressOK -->|Yes| RiskScore[Calculate Risk Score]
    
    RiskScore --> FinalDecision{Risk Score Acceptable?}
    FinalDecision -->|No| RejectScore[Reject - High Risk Score]
    FinalDecision -->|Yes| ApproveOrder[Approve Order]
    
    RejectPosition --> LogRejection[Log Rejection Reason]
    RejectVaR --> LogRejection
    RejectMargin --> LogRejection
    RejectCorr --> LogRejection
    RejectStress --> LogRejection
    RejectScore --> LogRejection
    
    LogRejection --> NotifyRiskTeam[Notify Risk Team]
    NotifyRiskTeam --> End([End])
    
    ApproveOrder --> SetRiskParams[Set Risk Parameters]
    SetRiskParams --> AttachStopLoss[Attach Stop Loss]
    AttachStopLoss --> End
```

## Options Strategy Flow

```mermaid
flowchart TD
    Start([Options Strategy Start]) --> MarketScan[Scan Options Market]
    MarketScan --> LoadChains[Load Option Chains]
    LoadChains --> CalcGreeks[Calculate Greeks]
    
    CalcGreeks --> StrategySelect{Select Strategy}
    
    StrategySelect --> IronCondor[Iron Condor]
    StrategySelect --> Butterfly[Butterfly Spread]
    StrategySelect --> Calendar[Calendar Spread]
    StrategySelect --> Straddle[Straddle/Strangle]
    StrategySelect --> Custom[Custom Multi-Leg]
    
    IronCondor --> CondorParams[Set IC Parameters]
    CondorParams --> FindStrikes[Find Optimal Strikes]
    
    Butterfly --> ButterflyParams[Set Butterfly Parameters]
    ButterflyParams --> FindStrikes
    
    Calendar --> CalendarParams[Set Calendar Parameters]
    CalendarParams --> FindExpiries[Find Optimal Expiries]
    
    Straddle --> StraddleParams[Set Straddle Parameters]
    StraddleParams --> FindATM[Find ATM Strike]
    
    Custom --> CustomBuilder[Strategy Builder]
    CustomBuilder --> ValidateLegs[Validate Leg Combination]
    
    FindStrikes --> VolAnalysis[Volatility Surface Analysis]
    FindExpiries --> VolAnalysis
    FindATM --> VolAnalysis
    ValidateLegs --> VolAnalysis
    
    VolAnalysis --> SkewCheck{Check Volatility Skew}
    SkewCheck --> TermStructure[Analyze Term Structure]
    TermStructure --> OpportunityScore[Calculate Opportunity Score]
    
    OpportunityScore --> ScoreThreshold{Score > Threshold?}
    ScoreThreshold -->|No| NextStrategy[Try Next Strategy]
    NextStrategy --> StrategySelect
    
    ScoreThreshold -->|Yes| BuildOrder[Build Spread Order]
    BuildOrder --> LegOptimization[Optimize Leg Execution]
    
    LegOptimization --> ExecutionAlgo{Execution Algorithm}
    ExecutionAlgo --> Simultaneous[Simultaneous Legs]
    ExecutionAlgo --> Sequential[Sequential Legs]
    ExecutionAlgo --> Legging[Legging In]
    
    Simultaneous --> CreateOrders[Create Multi-Leg Order]
    Sequential --> CreateOrders
    Legging --> CreateOrders
    
    CreateOrders --> ValidateSpread[Validate Spread Pricing]
    ValidateSpread --> CheckArbitrage{Arbitrage Opportunity?}
    
    CheckArbitrage -->|Yes| FastExecution[Priority Execution]
    CheckArbitrage -->|No| NormalExecution[Normal Execution]
    
    FastExecution --> SubmitOrder[Submit to Exchange]
    NormalExecution --> SubmitOrder
    
    SubmitOrder --> MonitorFills[Monitor Fill Status]
    MonitorFills --> AllFilled{All Legs Filled?}
    
    AllFilled -->|No| HandlePartial[Handle Partial Fills]
    HandlePartial --> UnwindDecision{Unwind or Complete?}
    UnwindDecision -->|Unwind| UnwindPosition[Unwind Filled Legs]
    UnwindDecision -->|Complete| RetryRemaining[Retry Remaining Legs]
    RetryRemaining --> MonitorFills
    
    AllFilled -->|Yes| RecordPosition[Record Spread Position]
    UnwindPosition --> RecordResult[Record Result]
    RecordPosition --> SetGreekLimits[Set Greek Limits]
    
    SetGreekLimits --> MonitorGreeks[Monitor Greeks]
    MonitorGreeks --> AdjustmentCheck{Need Adjustment?}
    
    AdjustmentCheck -->|Yes| PlanAdjustment[Plan Position Adjustment]
    PlanAdjustment --> StrategySelect
    
    AdjustmentCheck -->|No| ContinueMonitor[Continue Monitoring]
    ContinueMonitor --> End([End])
    RecordResult --> End
```

## Data Processing Flow

```mermaid
flowchart TD
    Start([Data Processing Start]) --> DataSources{Data Sources}
    
    DataSources --> AlpacaWS[Alpaca WebSocket]
    DataSources --> AlpacaREST[Alpaca REST API]
    DataSources --> MinIOHist[MinIO Historical]
    DataSources --> CustomFeeds[Custom Data Feeds]
    
    AlpacaWS --> StreamHandler[Stream Handler]
    AlpacaREST --> BatchHandler[Batch Handler]
    MinIOHist --> HistHandler[Historical Handler]
    CustomFeeds --> CustomHandler[Custom Handler]
    
    StreamHandler --> DataValidator[Data Validation]
    BatchHandler --> DataValidator
    HistHandler --> DataValidator
    CustomHandler --> DataValidator
    
    DataValidator --> QualityChecks{Quality Checks}
    QualityChecks --> Completeness[Check Completeness]
    QualityChecks --> Accuracy[Check Accuracy]
    QualityChecks --> Timeliness[Check Timeliness]
    QualityChecks --> Consistency[Check Consistency]
    
    Completeness --> QualityScore[Quality Score]
    Accuracy --> QualityScore
    Timeliness --> QualityScore
    Consistency --> QualityScore
    
    QualityScore --> PassQuality{Pass Quality?}
    PassQuality -->|No| DataCleaning[Data Cleaning]
    DataCleaning --> Interpolation[Missing Value Interpolation]
    DataCleaning --> OutlierDetection[Outlier Detection]
    DataCleaning --> ErrorCorrection[Error Correction]
    
    Interpolation --> CleanedData[Cleaned Data]
    OutlierDetection --> CleanedData
    ErrorCorrection --> CleanedData
    CleanedData --> DataValidator
    
    PassQuality -->|Yes| Normalization[Data Normalization]
    Normalization --> TimeAlignment[Time Alignment]
    TimeAlignment --> Aggregation[Data Aggregation]
    
    Aggregation --> TimeFrames{Timeframes}
    TimeFrames --> Tick[Tick Data]
    TimeFrames --> Second[1-Second Bars]
    TimeFrames --> Minute[1-Minute Bars]
    TimeFrames --> Hour[Hourly Bars]
    TimeFrames --> Daily[Daily Bars]
    
    Tick --> Storage[Storage Layer]
    Second --> Storage
    Minute --> Storage
    Hour --> Storage
    Daily --> Storage
    
    Storage --> StorageType{Storage Type}
    StorageType --> RedisCache[Redis Cache]
    StorageType --> PostgreSQL[PostgreSQL]
    StorageType --> MongoDB[MongoDB]
    StorageType --> MinIO[MinIO Archive]
    
    RedisCache --> UpdateCache[Update Cache]
    PostgreSQL --> UpdateDB[Update Database]
    MongoDB --> UpdateNoSQL[Update NoSQL]
    MinIO --> UpdateArchive[Update Archive]
    
    UpdateCache --> BroadcastUpdate[Broadcast Updates]
    UpdateDB --> BroadcastUpdate
    UpdateNoSQL --> BroadcastUpdate
    UpdateArchive --> BroadcastUpdate
    
    BroadcastUpdate --> Subscribers{Notify Subscribers}
    Subscribers --> TradingEngine[Trading Engine]
    Subscribers --> RiskManager[Risk Manager]
    Subscribers --> MLModels[ML Models]
    Subscribers --> GUI[User Interface]
    
    TradingEngine --> End([End])
    RiskManager --> End
    MLModels --> End
    GUI --> End
```

## AI Algorithm Decision Flow

```mermaid
flowchart TD
    Start([AI Decision Start]) --> MarketState[Capture Market State]
    MarketState --> ParallelAI{Run AI Algorithms in Parallel}
    
    ParallelAI --> Quantum[Quantum Analysis]
    ParallelAI --> Swarm[Swarm Intelligence]
    ParallelAI --> NAS[Neural Architecture Search]
    ParallelAI --> MetaRL[Meta-RL]
    ParallelAI --> Adversarial[Adversarial Prediction]
    
    Quantum --> QuantumCalc[Quantum State Calculation]
    QuantumCalc --> Superposition[Market Superposition]
    Superposition --> Entanglement[Asset Entanglement]
    Entanglement --> Tunneling[Tunneling Probability]
    Tunneling --> QuantumSignal[Quantum Signal]
    
    Swarm --> InitAgents[Initialize 100+ Agents]
    InitAgents --> AgentTypes{Agent Strategies}
    AgentTypes --> Scout[Scout Agents]
    AgentTypes --> Follower[Follower Agents]
    AgentTypes --> Contrarian[Contrarian Agents]
    Scout --> SwarmConsensus[Swarm Consensus]
    Follower --> SwarmConsensus
    Contrarian --> SwarmConsensus
    SwarmConsensus --> SwarmSignal[Swarm Signal]
    
    NAS --> ArchitectureEvolution[Evolve Architecture]
    ArchitectureEvolution --> FitnessEval[Evaluate Fitness]
    FitnessEval --> Selection[Select Best Architecture]
    Selection --> NASPrediction[NAS Prediction]
    NASPrediction --> NASSignal[NAS Signal]
    
    MetaRL --> LoadExperience[Load Meta-Experience]
    LoadExperience --> FastAdaptation[Fast Adaptation]
    FastAdaptation --> PolicyUpdate[Update Policy]
    PolicyUpdate --> MetaRLSignal[Meta-RL Signal]
    
    Adversarial --> GenerateScenarios[Generate Adversarial Scenarios]
    GenerateScenarios --> StressTest[Stress Test Predictions]
    StressTest --> RobustPrediction[Robust Prediction]
    RobustPrediction --> AdversarialSignal[Adversarial Signal]
    
    QuantumSignal --> SignalAggregation[Signal Aggregation]
    SwarmSignal --> SignalAggregation
    NASSignal --> SignalAggregation
    MetaRLSignal --> SignalAggregation
    AdversarialSignal --> SignalAggregation
    
    SignalAggregation --> WeightedConsensus[Weighted Consensus]
    WeightedConsensus --> ConfidenceCalc[Calculate Confidence]
    
    ConfidenceCalc --> ConfidenceLevel{Confidence Level}
    ConfidenceLevel -->|Low < 60%| NoTrade[No Trade Signal]
    ConfidenceLevel -->|Medium 60-80%| ConservativeTrade[Conservative Trade]
    ConfidenceLevel -->|High > 80%| AggressiveTrade[Aggressive Trade]
    
    NoTrade --> RecordDecision[Record Decision]
    ConservativeTrade --> PositionSizing[Calculate Position Size]
    AggressiveTrade --> PositionSizing
    
    PositionSizing --> KellyCalc[Kelly Criterion]
    KellyCalc --> RiskAdjust[Risk Adjustment]
    RiskAdjust --> FinalSignal[Final AI Signal]
    
    FinalSignal --> SignalValidation{Validate Signal}
    SignalValidation -->|Invalid| RejectSignal[Reject Signal]
    SignalValidation -->|Valid| AcceptSignal[Accept Signal]
    
    RejectSignal --> RecordDecision
    AcceptSignal --> RecordDecision
    RecordDecision --> UpdateModels[Update AI Models]
    UpdateModels --> End([End])
```

## Market Event Response Flow

```mermaid
flowchart TD
    Start([Market Event]) --> EventDetection[Event Detection System]
    EventDetection --> EventType{Event Type?}
    
    EventType --> NewsEvent[News Event]
    EventType --> EarningsEvent[Earnings Release]
    EventType --> EconomicData[Economic Data]
    EventType --> TechnicalEvent[Technical Breakout]
    EventType --> AnomalyEvent[Market Anomaly]
    
    NewsEvent --> NewsSentiment[Sentiment Analysis]
    NewsSentiment --> NewsImpact[Assess Impact]
    
    EarningsEvent --> EarningsAnalysis[Earnings Analysis]
    EarningsAnalysis --> CompareEstimates[Compare to Estimates]
    
    EconomicData --> MacroAnalysis[Macro Analysis]
    MacroAnalysis --> PolicyImplications[Policy Implications]
    
    TechnicalEvent --> PatternRecognition[Pattern Recognition]
    PatternRecognition --> TechnicalSignals[Generate Technical Signals]
    
    AnomalyEvent --> AnomalyAnalysis[Anomaly Analysis]
    AnomalyAnalysis --> RiskAssessment[Risk Assessment]
    
    NewsImpact --> ResponseStrategy[Determine Response Strategy]
    CompareEstimates --> ResponseStrategy
    PolicyImplications --> ResponseStrategy
    TechnicalSignals --> ResponseStrategy
    RiskAssessment --> ResponseStrategy
    
    ResponseStrategy --> ResponseType{Response Type}
    
    ResponseType --> Defensive[Defensive Response]
    ResponseType --> Neutral[Neutral Response]
    ResponseType --> Aggressive[Aggressive Response]
    ResponseType --> Emergency[Emergency Response]
    
    Defensive --> ReduceExposure[Reduce Exposure]
    ReduceExposure --> HedgePositions[Hedge Positions]
    
    Neutral --> MonitorOnly[Monitor Only]
    MonitorOnly --> SetAlerts[Set Alerts]
    
    Aggressive --> IncreasePositions[Increase Positions]
    IncreasePositions --> AddToWinners[Add to Winners]
    
    Emergency --> CircuitBreaker[Activate Circuit Breaker]
    CircuitBreaker --> CloseAllPositions[Close All Positions]
    
    HedgePositions --> ExecuteResponse[Execute Response]
    SetAlerts --> ExecuteResponse
    AddToWinners --> ExecuteResponse
    CloseAllPositions --> ExecuteResponse
    
    ExecuteResponse --> MonitorOutcome[Monitor Outcome]
    MonitorOutcome --> UpdateRisk[Update Risk Parameters]
    UpdateRisk --> LogEvent[Log Event Response]
    LogEvent --> NotifyStakeholders[Notify Stakeholders]
    NotifyStakeholders --> End([End])
```

## System Integration Flow

```mermaid
flowchart TD
    Start([System Integration]) --> ExternalAPIs{External APIs}
    
    ExternalAPIs --> AlpacaAPI[Alpaca Trading API]
    ExternalAPIs --> MarketDataAPIs[Market Data Providers]
    ExternalAPIs --> EconomicAPIs[Economic Data APIs]
    ExternalAPIs --> CloudServices[Cloud Services]
    
    AlpacaAPI --> APIGateway[API Gateway]
    MarketDataAPIs --> APIGateway
    EconomicAPIs --> APIGateway
    CloudServices --> APIGateway
    
    APIGateway --> Authentication[Authentication Layer]
    Authentication --> RateLimiting[Rate Limiting]
    RateLimiting --> RequestValidation[Request Validation]
    
    RequestValidation --> ServiceRouter[Service Router]
    ServiceRouter --> ServiceType{Service Type}
    
    ServiceType --> TradingService[Trading Service]
    ServiceType --> DataService[Data Service]
    ServiceType --> AnalyticsService[Analytics Service]
    ServiceType --> RiskService[Risk Service]
    
    TradingService --> OrderManager[Order Manager]
    DataService --> DataManager[Data Manager]
    AnalyticsService --> MLPipeline[ML Pipeline]
    RiskService --> RiskEngine[Risk Engine]
    
    OrderManager --> MessageQueue[Message Queue]
    DataManager --> MessageQueue
    MLPipeline --> MessageQueue
    RiskEngine --> MessageQueue
    
    MessageQueue --> EventProcessing[Event Processing]
    EventProcessing --> EventStore[Event Store]
    EventStore --> StreamProcessing[Stream Processing]
    
    StreamProcessing --> RealtimeAnalytics[Real-time Analytics]
    RealtimeAnalytics --> Dashboard[Dashboard Updates]
    Dashboard --> WebSocket[WebSocket Server]
    WebSocket --> ClientApps[Client Applications]
    
    ClientApps --> GUI[Trading GUI]
    ClientApps --> MobileApp[Mobile App]
    ClientApps --> RESTAPI[REST API Clients]
    
    GUI --> End([End])
    MobileApp --> End
    RESTAPI --> End
```

---

These flow charts represent the complete operational flows of the Alpaca-MCP trading system, showing how data moves through the system, how decisions are made, and how various components interact to execute trades and manage risk in real-time.