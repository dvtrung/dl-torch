import {Button, Icon, Row, Typography, Col} from "antd";
import Stats from "../Stats";
import Terminal from "../Terminal";
import {syncEpochStats, syncEpochStepStats} from "../../actions/home";
import {connect} from "react-redux";
import React, {Component} from "react";
import type {Machine, Model} from "../../reducers/types";
import {bindActionCreators} from "redux";
import {Line, Scatter} from 'react-chartjs-2';

const { Text } = Typography;

type Props = {
  model: Model,
  machine: Machine,
  syncEpochStats: (Model, Machine, boolean) => void,
  syncEpochStepStats: (Model, Machine) => void
};

class TrainingTab extends Component<Props> {
  constructor() {
    super();
    this.state = {
      epoch: null,
      metrics: [null],
    }
  }

  handleRefreshStats = () => {
    this.props.syncEpochStats(this.props.model, this.props.machine, true);
    this.props.syncEpochStepStats(this.props.model, this.props.machine);
  }

  render() {
    const { model, machine } = this.props;
    return (
      <div>
        <Row>
          <Stats stats={model.stats} stepStats={model.stepStats} />
          <Text type="secondary">Last updated: {model.stats.lastUpdated ? model.stats.lastUpdated.toLocaleTimeString() : "never"}</Text>
          <Button type="link" onClick={this.handleRefreshStats} disabled={model.stats.isLoading}>
            {this.props.model.stats.isLoading ? <Icon type="sync" spin /> : <Icon type="sync" />}
          </Button>
          <Terminal machine={this.props.machine}/>
        </Row>
        <Row>
          <Col span={12}>
            <Line
              data={{
                labels: model.stats.metrics ? model.stats.epochs : [],
                datasets: [
                  {
                    label: "Train loss",
                    data: model.stats.metrics ? model.stats.results[0] : [],
                    fill: false,
                    lineTension: 0
                  }
                  ]
              }}
              options={{
                datasetFill: false,
                bezierCurve: false,
              }}
            />
          </Col>
          <Col span={12}>
            <Scatter
              data={{
                datasets: [
                  {
                    label: "Train loss",
                    data: model.stepStats.losses.map((loss, i) => ({
                      x: model.stepStats.epochs[i],
                      y: loss
                    })),
                    showLine: true,
                    fill: false,
                    lineTension: 0
                  }]
              }}
              options={{
                datasetFill: false,
                bezierCurve: false,
                scales: {
                  xAxes: [{
                    display: true,
                    scaleLabel: {
                      display: true,
                      labelString: 'epoch',
                    },
                    ticks: {
                      callback: (value, index, values) => parseFloat(value).toFixed(2),
                      autoSkip: true,
                      stepSize: .05
                    }
                  }],
                }
              }}
            />
          </Col>
        </Row>
      </div>
    )
  }
}

function mapStateToProps(state) {
  const model = state.home.models[state.home.selectedModelKey];
  return {
    model,
    machine: state.settings.machines[model.machine]
  };
}

function mapDispatchToProps(dispatch) {
  return {
    syncEpochStats: bindActionCreators(syncEpochStats, dispatch),
    syncEpochStepStats: bindActionCreators(syncEpochStepStats, dispatch)
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingTab);
