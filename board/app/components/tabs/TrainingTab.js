import {Button, Icon, Row} from "antd";
import Stats from "../Stats";
import Terminal from "../Terminal";
import {syncTrainingStats} from "../../actions/home";
import {connect} from "react-redux";
import React, {Component} from "react";
import type {Machine, Model} from "../../reducers/types";

type Props = {
  model: Model,
  machine: Machine,
  syncTrainingStats: (Model, Machine) => void
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
    console.log(this.props.syncTrainingStats)
    this.props.syncTrainingStats(this.props.model, this.props.machine);
  }

  render() {
    const { model, machine } = this.props;
    return (
      <Row>
        {model.stats.epoch && <Stats stats={model.stats} />}
        <Button onClick={this.handleRefreshStats} disabled={this.props.model.stats.isLoading}>
          {this.props.model.stats.isLoading && <Icon type="sync" spin />} Refresh
        </Button>
        <Terminal machine={this.props.machine}/>
      </Row>
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
    syncTrainingStats: (model, machine) => syncTrainingStats(model, machine)(dispatch)
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingTab);
