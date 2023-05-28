import React from 'react';
import styles from './Client.module.scss';
import { useNavigate } from 'react-router-dom';
import AllObjects from '../../components/AllObjects/AllObjects';

const Client = () => {
	const navigate = useNavigate();

	const selectObject = (id) => {
		navigate(`/client/${id}`);
	};

	return (
		<AllObjects title={'Выберите ЖК'} handleSelectObject={(id) => selectObject(id)}/>
	);
};

export default Client;