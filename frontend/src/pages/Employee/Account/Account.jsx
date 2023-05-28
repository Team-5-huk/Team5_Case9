import React, { useEffect, useState } from 'react';
import styles from './Account.module.scss';
import CustomButton from '../../../components/CustomButton/CustomButton';
import store from '../../../store/store';
import ListCompleted from '../../../components/ListCompleted/ListCompleted';
import linksStore from '../../../store/linksStore';
import { apiGetUser } from '../../../api/api';

const rounds = [
	{id: 1, date: '01.05.2022', jc: 'ЖК No1', frame: '1', apartaments: '20'},
	{id: 2, date: '02.05.2022', jc: 'ЖК No1', frame: '1', apartaments: '20'},
	{id: 3, date: '03.05.2022', jc: 'ЖК No2', frame: '1', apartaments: '20'},
	{id: 4, date: '04.05.2022', jc: 'ЖК No2', frame: '1', apartaments: '20'},
	{id: 5, date: '05.05.2022', jc: 'ЖК No3', frame: '1', apartaments: '20'},
	{id: 6, date: '06.05.2022', jc: 'ЖК No3', frame: '1', apartaments: '20'},
	{id: 7,date: '07.05.2022', jc: 'ЖК No4', frame: '1', apartaments: '20'},

];

const columns = [
	{ field: 'date', headerName: 'Дата', flex: 1 },
	{ field: 'jc', headerName: 'ЖК', flex: 1 },
	{ field: 'frame', headerName: 'Корпус', flex: 1 },
	{ field: 'apartaments', headerName: 'Кол-во квартир', flex: 1 },
];

const Account = () => {
	const { setAuthUser } = store;
	const { linkGetUser } = linksStore;

	const [user, setUser] = useState({});

	useEffect(() => {
		getInfoUser();
	}, []);

	const getInfoUser = () => {
		apiGetUser(linkGetUser).then(({ data, error }) => {
			setUser(data);
			console.log(error);
		});
	};

	const signOut = () => {
		localStorage.removeItem('token');
		localStorage.removeItem('role');
		setAuthUser('');
	};

	return (
		<div className={styles.container}>
			<div>
				{user.first_name} {user.last_name}
				{user.email}
				<CustomButton name={'Выйти'} handleClick={signOut} />
			</div>

			Мои обходы:
			<ListCompleted columns={columns} data={rounds} />

		</div>
	);
};

export default Account;
