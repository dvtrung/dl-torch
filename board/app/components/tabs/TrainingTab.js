import {Icon, Row, Typography, Col, Progress, Button, Statistic, Tooltip, Card} from "antd";
import {syncEpochStats, syncEpochStepStats} from "../../actions/home";
import {connect} from "react-redux";
import React, {Component} from "react";
import type {Machine, Model} from "../../reducers/types";
import {bindActionCreators} from "redux";
import {Scatter} from 'react-chartjs-2';

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
    if (!this.props.model.stepStats.isConnected) {
      this.props.syncEpochStepStats(this.props.model, this.props.machine);
    }
  };

  render() {
    const { model, machine } = this.props;
    const connectBtn = model.stepStats.isConnected ?
      <Button type="danger">Disconnect</Button> :
      (model.stepStats.isConnecting ?
          <Button disabled={true}>Connecting...</Button> :
          <Button type="primary" onClick={this.handleRefreshStats}>Connect</Button>
      )
    const machineStatusText = model.stepStats.isConnected ?
        <Text type="warning"><Icon type="check" /> Connected</Text> :
        (model.stepStats.isConnecting ?
          <Text type="secondary"><Icon type="sync" spin /> Connecting...</Text> :
          <Text type="danger"><Icon type="close" /> Not connected</Text>)
    return (
      <div>
        <Row>
          <Text>Machine: </Text>{connectBtn}
        </Row>
        <Row gutter={16} type="flex">
          <Col span={8}>
            <Card style={{textAlign: "center", height: "100%"}}>
              <h3>Epoch Progress</h3>
              <Progress type="circle" percent={Math.floor(model.stepStats.epoch * 100) % 100} />
            </Card>
          </Col>
          <Col span={8}>
            <Card
              style={{textAlign: "center", height: "100%"}}
              actions={[]}
            >
              <div style={{height: 110}}>
                <h3>Machine Status</h3>
              </div>
              </Card>
          </Col>
          <Col span={8}>
            <Card
              style={{textAlign: "center", height: "100%"}}
              actions={[<span>Train</span>]}>
              <div style={{height: 110}}>
                <h3>Model Status</h3>
              </div>
            </Card>
          </Col>
        </Row>
        <Row>
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
                interactive: true,
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
