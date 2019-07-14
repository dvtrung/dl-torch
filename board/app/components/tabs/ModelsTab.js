import {Button, Icon, Row, Typography, Col, Card} from "antd";
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

class ModelsTab extends Component<Props> {
  constructor() {
    super();
    this.state = {
      epoch: null,
      metrics: [null],
    }
  }

  handleRefreshStats = () => {
    this.props.syncEpochStats(this.props.model, this.props.machine, true);
  };

  render() {
    const { machine } = this.props;
    return (
      <div>
        {Object.entries(machine.models).map(([key, model]) => (
          <Card key={key}>
            <h4><a>{key}</a></h4>
            <Text type="secondary">{model.localPath}</Text>
          </Card>
        ))}
      </div>
    )
  }
}

function mapStateToProps(state) {
  const machine = state.home.machines[state.home.selectedMachineKey];
  return {
    machine,
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
)(ModelsTab);
