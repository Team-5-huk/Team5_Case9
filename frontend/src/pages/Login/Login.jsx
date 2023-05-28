import React, { useState } from 'react';
import styles from './Login.module.scss';
import store from '../../store/store';
import linksStore from '../../store/linksStore';
import CustomInput from '../../components/CustomInput/CustomInput';
import CustomButton from '../../components/CustomButton/CustomButton';
import { apiPostAuthorize } from '../../api/api';
import { observer } from 'mobx-react-lite';

const Login = observer(() => {
	const {setAuthUser} = store;
	const {linkLogin} = linksStore;

	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');

	const loginApi = () => {
		apiPostAuthorize(linkLogin, {email, password}).then(({data, error}) => {
				console.log(error);
				setAuthUser(data.role);
				localStorage.setItem('role', data.role);
				localStorage.setItem('token', data.token);
		})
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>Войти в аккаунт</div>

			<div className={styles.border}/>

			<div className={styles.input}>
				<CustomInput 
					type="email"
					placeholder="email"
					value={email}
					onChange={(e) => setEmail(e)}
				/>

				<CustomInput 
					type="password"
					placeholder="пароль"
					value={password}
					onChange={(e) => setPassword(e)}
				/>
			</div>

			<div style={{width:'90%'}}>
				<CustomButton name='Войти' handleClick={loginApi}/>
			</div>
		</div>
	);
});

export default Login;
