// @flow
import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import routes from '../constants/routes';
import styles from './Home.css';
import { Layout, Button, Input, Row, Col, Tabs } from 'antd';
import Stats from './Stats'
import Tree from './Tree'
import MachineSelect from './MachineSelect'
import {loadSettings, default as SettingActions} from '../actions/settings';
import type {Machine} from "../reducers/types";
import {bindActionCreators} from "redux";
import * as HomeActions from "../actions/home";
import {connect} from "react-redux";
import Home from "./Home";
const { Sider, Content } = Layout;
const { TextArea } = Input;
const { TabPane } = Tabs;

type Props = {
  machine: any,
  sshConnect: (machine: Machine) => void
};

class Terminal extends Component<Props> {
  constructor(props) {
    super(props)
  }

  componentDidMount() {

  }

  onConnect = () => {
    this.props.sshConnect(this.props.machine)
  };

  render() {
    return (
      <Button type="primary" onClick={this.onConnect}>Connect</Button>
    );
  }
}

function mapStateToProps(state) {
  return {

  };
}

const mapDispatchToProps = dispatch => {
  return {
    // dispatching plain actions
    sshConnect: () => dispatch({ type: 'INCREMENT' }),
    decrement: () => dispatch({ type: 'DECREMENT' }),
    reset: () => dispatch({ type: 'RESET' })
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Terminal);

