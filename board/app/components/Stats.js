// @flow
import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import routes from '../constants/routes';
import styles from './Home.css';
import { Statistic, Row, Col, Icon } from 'antd';
type Props = {
  epoch: number,
  totalEpoch: number,
  trainError: number,
  testError: number,
};

export default class Stats extends Component<Props> {
  onSelect = (keys, event) => {
    console.log('Trigger Select', keys, event);
  };

  onExpand = () => {
    console.log('Trigger Expand');
  };

  render() {
    return (
        <Row gutter={16}>
    <Col span={8}>
      <Statistic title="Epoch" value={this.props.epoch} suffix={"/ " + this.props.totalEpoch} />
    </Col>
    <Col span={8}>
      <Statistic title="Train Error" value={`${this.props.trainError}%`} />
    </Col>
    <Col span={8}>
      <Statistic title="Test Error" value={`${this.props.testError}%`} />
    </Col>
  </Row>
    );
  }
}

